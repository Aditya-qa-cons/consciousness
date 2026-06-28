"""Unit tests for the knowledge graph builder and DB methods."""

import pytest

from consciousness.memory.knowledge_graph import KnowledgeGraphBuilder, _tech_id, _topic_id
from consciousness.models import Decision, KGEdge, KGNode, TechChoice
from tests.conftest import make_conversation, make_project, utc

# ── fixtures ──────────────────────────────────────────────────────────────────


_tc_seq = 0


def _seed_tech(db, technology, verdict, conversation_id="conv-1"):
    global _tc_seq
    _tc_seq += 1
    tc = TechChoice(
        id=f"tc-{_tc_seq}",
        technology=technology,
        verdict=verdict,
        conversation_id=conversation_id,
        extracted_at=utc(2024, 1, 1),
    )
    db.upsert_tech_choice(tc)
    return tc


def _seed_decision(db, id_, topic, conclusion, conversation_id="conv-1", superseded_by=None):
    d = Decision(
        id=id_,
        topic=topic,
        conclusion=conclusion,
        conversation_id=conversation_id,
        extracted_at=utc(2024, 1, 1),
        superseded_by=superseded_by,
    )
    db.upsert_decision(d)
    return d


@pytest.fixture
def seeded(db):
    """DB pre-loaded with projects and conversations needed for FK constraints."""
    db.upsert_project(make_project())
    db.upsert_conversation(make_conversation(id="conv-1"))
    db.upsert_conversation(make_conversation(id="conv-2", title="Auth strategy"))
    db.commit()
    return db


# ── node creation ─────────────────────────────────────────────────────────────


def test_builder_creates_tech_nodes(seeded):
    _seed_tech(seeded, "PostgreSQL", "Use for production")
    _seed_tech(seeded, "Redis", "Use for caching")
    seeded.commit()

    nodes, _ = KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    assert nodes >= 2
    pg = seeded.get_kg_node(_tech_id("PostgreSQL"))
    assert pg is not None
    assert pg.type == "technology"
    assert pg.label == "PostgreSQL"


def test_builder_deduplicates_tech_nodes(seeded):
    _seed_tech(seeded, "postgres", "v1", conversation_id="conv-1")
    _seed_tech(seeded, "Postgres", "v2", conversation_id="conv-2")
    seeded.commit()

    nodes, _ = KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    assert nodes == 1  # same normalized name


def test_builder_creates_topic_nodes(seeded):
    _seed_decision(seeded, "d1", "database choice", "Use Postgres for production")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    node = seeded.get_kg_node(_topic_id("database choice"))
    assert node is not None
    assert node.type == "topic"


# ── co-occurrence edges ───────────────────────────────────────────────────────


def test_builder_co_occurrence_edge(seeded):
    _seed_tech(seeded, "PostgreSQL", "Use for DB", conversation_id="conv-1")
    _seed_tech(seeded, "Redis", "Use for cache", conversation_id="conv-1")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    pairs = seeded.co_occurring_technologies()
    labels = {frozenset([t1, t2]) for t1, t2, _ in pairs}
    assert frozenset(["PostgreSQL", "Redis"]) in labels


def test_builder_co_occurrence_weight(seeded):
    # Same pair in two separate conversations
    _seed_tech(seeded, "PostgreSQL", "v1", conversation_id="conv-1")
    _seed_tech(seeded, "Redis", "v1", conversation_id="conv-1")
    _seed_tech(seeded, "PostgreSQL", "v2", conversation_id="conv-2")
    _seed_tech(seeded, "Redis", "v2", conversation_id="conv-2")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    pairs = seeded.co_occurring_technologies()
    assert pairs[0][2] == 2.0  # weight = number of conversations


def test_builder_no_co_occurrence_edge_single_tech(seeded):
    _seed_tech(seeded, "PostgreSQL", "Use for DB", conversation_id="conv-1")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    pairs = seeded.co_occurring_technologies()
    assert pairs == []


# ── superseded_by edges ───────────────────────────────────────────────────────


def test_builder_superseded_by_edge(seeded):
    # d2 must exist before d1 references it via superseded_by FK
    _seed_decision(seeded, "d2", "database choice v2", "Use PostgreSQL instead")
    _seed_decision(seeded, "d1", "database choice", "Use SQLite", superseded_by="d2")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    src = _topic_id("database choice")
    dst = _topic_id("database choice v2")
    neighbors = seeded.get_kg_neighbors(src, relation="superseded_by")
    dst_ids = [n.id for _, n in neighbors]
    assert dst in dst_ids


# ── relates_to edges ──────────────────────────────────────────────────────────


def test_builder_relates_to_edge(seeded):
    _seed_tech(seeded, "PostgreSQL", "Use for production")
    _seed_decision(seeded, "d1", "database choice", "We decided to use PostgreSQL for all services")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    topic_nid = _topic_id("database choice")
    tech_nid = _tech_id("PostgreSQL")
    neighbors = seeded.get_kg_neighbors(topic_nid, relation="relates_to")
    assert any(n.id == tech_nid for _, n in neighbors)


def test_builder_relates_to_reverse_lookup(seeded):
    """Tech node neighbors include topic nodes that relate to it."""
    _seed_tech(seeded, "Redis", "Use for cache")
    _seed_decision(seeded, "d1", "caching strategy", "Redis works best for session caching")
    seeded.commit()

    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    tech_nid = _tech_id("Redis")
    # get_kg_neighbors returns both directions, so topic should appear as incoming edge
    neighbors = seeded.get_kg_neighbors(tech_nid)
    topic_labels = [n.label for _, n in neighbors if n.type == "topic"]
    assert any("caching" in label.lower() for label in topic_labels)


# ── rebuild clears old data ───────────────────────────────────────────────────


def test_builder_clears_on_rebuild(seeded):
    _seed_tech(seeded, "PostgreSQL", "Use for DB")
    seeded.commit()
    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    # Remove the tech choice and rebuild — node should be gone
    seeded.conn.execute("DELETE FROM tech_choices")
    seeded.commit()
    nodes, edges = KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    assert nodes == 0
    assert edges == 0


# ── DB query methods ──────────────────────────────────────────────────────────


def test_db_co_occurring_technologies_sorted(seeded):
    _seed_tech(seeded, "A", "v", conversation_id="conv-1")
    _seed_tech(seeded, "B", "v", conversation_id="conv-1")
    _seed_tech(seeded, "A", "v2", conversation_id="conv-2")
    _seed_tech(seeded, "B", "v2", conversation_id="conv-2")
    _seed_tech(seeded, "C", "v", conversation_id="conv-1")
    seeded.commit()
    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    pairs = seeded.co_occurring_technologies()
    # A+B appears in 2 convs, A+C and B+C in 1 each
    assert pairs[0][2] >= pairs[1][2]  # sorted descending


def test_db_revisited_topics(seeded):
    _seed_decision(seeded, "d1", "database choice", "Use SQLite")
    _seed_decision(seeded, "d2", "DATABASE CHOICE", "Use Postgres")
    _seed_decision(seeded, "d3", "auth strategy", "Use JWT")
    seeded.commit()

    topics = seeded.revisited_topics()
    topic_names = [t.lower() for t, _ in topics]
    assert any("database choice" in name for name in topic_names)
    assert not any("auth strategy" in name for name in topic_names)


def test_db_get_kg_neighbors_outgoing(db):
    db.upsert_kg_node(KGNode(id="tech:pg", type="technology", label="PostgreSQL"))
    db.upsert_kg_node(KGNode(id="tech:redis", type="technology", label="Redis"))
    db.upsert_kg_edge(KGEdge(src_id="tech:pg", dst_id="tech:redis", relation="co_occurs_with", weight=3.0))
    db.commit()

    neighbors = db.get_kg_neighbors("tech:pg")
    assert len(neighbors) == 1
    edge, node = neighbors[0]
    assert node.label == "Redis"
    assert edge.weight == 3.0


def test_db_get_kg_neighbors_incoming(db):
    db.upsert_kg_node(KGNode(id="topic:db", type="topic", label="database choice"))
    db.upsert_kg_node(KGNode(id="tech:pg", type="technology", label="PostgreSQL"))
    db.upsert_kg_edge(KGEdge(src_id="topic:db", dst_id="tech:pg", relation="relates_to"))
    db.commit()

    # Query from tech node — should see the topic as an incoming neighbor
    neighbors = db.get_kg_neighbors("tech:pg")
    assert len(neighbors) == 1
    _, node = neighbors[0]
    assert node.type == "topic"


def test_db_get_kg_neighbors_relation_filter(db):
    db.upsert_kg_node(KGNode(id="tech:pg", type="technology", label="PostgreSQL"))
    db.upsert_kg_node(KGNode(id="tech:redis", type="technology", label="Redis"))
    db.upsert_kg_node(KGNode(id="topic:db", type="topic", label="DB"))
    db.upsert_kg_edge(KGEdge(src_id="tech:pg", dst_id="tech:redis", relation="co_occurs_with"))
    db.upsert_kg_edge(KGEdge(src_id="topic:db", dst_id="tech:pg", relation="relates_to"))
    db.commit()

    only_co = db.get_kg_neighbors("tech:pg", relation="co_occurs_with")
    assert len(only_co) == 1
    assert only_co[0][1].label == "Redis"


def test_db_stats_includes_kg(db):
    db.upsert_kg_node(KGNode(id="tech:pg", type="technology", label="PostgreSQL"))
    db.upsert_kg_node(KGNode(id="tech:redis", type="technology", label="Redis"))
    db.upsert_kg_edge(KGEdge(src_id="tech:pg", dst_id="tech:redis", relation="co_occurs_with", weight=1.0))
    db.commit()

    s = db.stats()
    assert s["kg_nodes"] == 2
    assert s["kg_edges"] == 1


# ── MCP handler ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_co_occurring_technologies(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    _seed_tech(seeded, "PostgreSQL", "Use it", conversation_id="conv-1")
    _seed_tech(seeded, "Redis", "Cache layer", conversation_id="conv-1")
    seeded.commit()
    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    result = await explore_knowledge_graph(seeded, {"query": "co_occurring_technologies"})
    text = result[0].text
    assert "PostgreSQL" in text or "Redis" in text


@pytest.mark.asyncio
async def test_mcp_revisited_topics(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    _seed_decision(seeded, "d1", "database choice", "Use SQLite")
    _seed_decision(seeded, "d2", "Database Choice", "Use Postgres")
    seeded.commit()

    result = await explore_knowledge_graph(seeded, {"query": "revisited_topics"})
    assert "database choice" in result[0].text.lower()


@pytest.mark.asyncio
async def test_mcp_technology_context(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    _seed_tech(seeded, "PostgreSQL", "Use for production")
    _seed_tech(seeded, "Redis", "Cache layer", conversation_id="conv-2")
    _seed_tech(seeded, "PostgreSQL", "Still prefer it", conversation_id="conv-2")
    seeded.commit()
    KnowledgeGraphBuilder().rebuild(seeded)
    seeded.commit()

    result = await explore_knowledge_graph(seeded, {"query": "technology_context", "technology": "PostgreSQL"})
    text = result[0].text
    assert "PostgreSQL" in text
    assert "Redis" in text  # co-occurs in conv-2


@pytest.mark.asyncio
async def test_mcp_technology_context_missing_node(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    result = await explore_knowledge_graph(seeded, {"query": "technology_context", "technology": "Nonexistent"})
    assert "No graph node found" in result[0].text


@pytest.mark.asyncio
async def test_mcp_co_occurring_no_data(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    result = await explore_knowledge_graph(seeded, {"query": "co_occurring_technologies"})
    assert "No co-occurrence data" in result[0].text


@pytest.mark.asyncio
async def test_mcp_unknown_query(seeded):
    from consciousness.mcp_server.server import explore_knowledge_graph

    result = await explore_knowledge_graph(seeded, {"query": "bad_query"})
    assert "Unknown query" in result[0].text
