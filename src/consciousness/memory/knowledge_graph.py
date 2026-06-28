"""Build the knowledge graph from extracted decisions and technology choices."""

import re
from collections import defaultdict

from consciousness.models import KGEdge, KGNode
from consciousness.store.db import Database


def _tech_id(label: str) -> str:
    return f"tech:{label.lower().strip()}"


def _topic_id(topic: str) -> str:
    return f"topic:{topic.lower().strip()}"


class KnowledgeGraphBuilder:
    """Derives graph nodes and edges from the decisions and tech_choices tables."""

    def rebuild(self, db: Database) -> tuple[int, int]:
        """Drop and recreate all graph data. Returns (node_count, edge_count)."""
        db.clear_kg()

        # ── 1. Technology nodes (deduplicated by normalized name) ─────────────
        tech_choices = db.list_tech_choices()
        canonical: dict[str, str] = {}   # lower → first-seen original label
        for tc in tech_choices:
            label = tc.technology.strip()
            if not label:
                continue
            key = label.lower()
            if key not in canonical:
                canonical[key] = label

        for key, label in canonical.items():
            db.upsert_kg_node(KGNode(id=_tech_id(label), type="technology", label=label))

        # ── 2. Topic nodes (active decisions, deduplicated by normalized topic) ─
        decisions = db.list_decisions(limit=10_000)
        canonical_topics: dict[str, str] = {}  # lower → first-seen original label
        for d in decisions:
            key = d.topic.lower().strip()
            if key not in canonical_topics:
                canonical_topics[key] = d.topic

        for label in canonical_topics.values():
            db.upsert_kg_node(KGNode(id=_topic_id(label), type="topic", label=label))

        # ── 3. Co-occurrence edges (techs sharing a conversation) ─────────────
        by_conv: dict[str, list[str]] = defaultdict(list)
        for tc in tech_choices:
            key = tc.technology.lower().strip()
            if key in canonical:
                by_conv[tc.conversation_id].append(key)

        co_counts: dict[tuple[str, str], int] = {}
        for techs in by_conv.values():
            uniq = list(set(techs))
            for i, t1 in enumerate(uniq):
                for t2 in uniq[i + 1:]:
                    pair = (min(t1, t2), max(t1, t2))
                    co_counts[pair] = co_counts.get(pair, 0) + 1

        for (t1, t2), count in co_counts.items():
            db.upsert_kg_edge(KGEdge(
                src_id=_tech_id(t1), dst_id=_tech_id(t2),
                relation="co_occurs_with", weight=float(count),
            ))

        # ── 4. Superseded_by edges ────────────────────────────────────────────
        all_decisions = db.list_all_decisions()
        d_by_id = {d.id: d for d in all_decisions}

        for d in all_decisions:
            if not d.superseded_by or d.superseded_by not in d_by_id:
                continue
            src_label = d.topic
            dst_label = d_by_id[d.superseded_by].topic
            src = _topic_id(src_label)
            dst = _topic_id(dst_label)
            if src == dst:
                continue
            # Superseded decisions may not have active topic nodes — ensure they exist.
            if not db.get_kg_node(src):
                db.upsert_kg_node(KGNode(id=src, type="topic", label=src_label))
            if not db.get_kg_node(dst):
                db.upsert_kg_node(KGNode(id=dst, type="topic", label=dst_label))
            db.upsert_kg_edge(KGEdge(src_id=src, dst_id=dst, relation="superseded_by"))

        # ── 5. Relates_to edges (tech names appearing in decision text) ───────
        if canonical:
            pattern = re.compile(
                r'\b(' + '|'.join(
                    re.escape(k) for k in sorted(canonical, key=len, reverse=True)
                ) + r')\b',
                re.IGNORECASE,
            )
            for d in decisions:
                topic_nid = _topic_id(d.topic)
                text = f"{d.topic} {d.conclusion}"
                matched: set[str] = set()
                for m in pattern.finditer(text):
                    tech_key = m.group(1).lower()
                    if tech_key not in matched:
                        matched.add(tech_key)
                        db.upsert_kg_edge(KGEdge(
                            src_id=topic_nid,
                            dst_id=_tech_id(canonical[tech_key]),
                            relation="relates_to",
                        ))

        row = db.conn.execute(
            "SELECT (SELECT COUNT(*) FROM kg_nodes) AS n, (SELECT COUNT(*) FROM kg_edges) AS e"
        ).fetchone()
        return row["n"], row["e"]
