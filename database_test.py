"""Run: python -m pytest -s database_test.py"""

import pytest
from database import Database, Table


class TestTable(Table):
    id: int
    name: str
    value: float
    is_active: bool

    @classmethod
    def primary_key(cls) -> tuple[str]:
        return ("id INTEGER",)


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def setup_table(db):
    db.create_table(TestTable, "test_table", index_columns=["name", "is_active"])
    return db


def test_create_table(db):
    db.create_table(TestTable, "test_table", index_columns=["name", "is_active"])
    assert db.count("test_table") == 0


def test_add_and_get(setup_table):
    test_item = TestTable(id=1, name="test", value=10.5, is_active=True)
    setup_table.add("test_table", test_item)

    result = setup_table.get("test_table")
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["name"] == "test"
    assert result[0]["value"] == 10.5
    assert result[0]["is_active"] is True


# TODO: Duplicate primary key exceptions are ignored for now... rely on count checks from `add` instead.
# def test_add_duplicate_primary_key(setup_table):
#     dup_id = 1
#     item1 = TestTable(id=dup_id, name="test1", value=10.5, is_active=True)
#     item2 = TestTable(id=dup_id, name="test2", value=81.9, is_active=False)
#     setup_table.add("test_table", item1)
#     with pytest.raises(Exception):  # Expecting a constraint violation
#        setup_table.add("test_table", item2)


def test_get_with_condition(setup_table):
    setup_table.add(
        "test_table", TestTable(id=1, name="active1", value=10.5, is_active=True)
    )
    setup_table.add(
        "test_table", TestTable(id=2, name="inactive", value=20.5, is_active=False)
    )
    setup_table.add(
        "test_table", TestTable(id=3, name="active2", value=30.5, is_active=True)
    )

    result = setup_table.get("test_table", condition="is_active = TRUE")
    assert len(result) == 2
    assert all(item["is_active"] for item in result)

    result = setup_table.get("test_table", condition="name LIKE 'active%'")
    assert len(result) == 2
    assert all(item["name"].startswith("active") for item in result)

    result = setup_table.get("test_table", condition="is_active = TRUE AND value > 20")
    assert len(result) == 1
    assert result[0]["name"] == "active2"


def test_sample(setup_table):
    for i in range(10):
        setup_table.add(
            "test_table",
            TestTable(id=i, name=f"test{i}", value=float(i), is_active=i % 2 == 0),
        )

    result = setup_table.sample("test_table", k=5)
    assert len(result) == 5
    assert all(isinstance(item["id"], int) for item in result)

    cond_result = setup_table.sample("test_table", k=2, condition="is_active = True")
    assert len(cond_result) == 2


def test_count(setup_table):
    assert setup_table.count("test_table") == 0

    setup_table.add(
        "test_table", TestTable(id=1, name="test1", value=10.5, is_active=True)
    )
    assert setup_table.count("test_table") == 1

    setup_table.add(
        "test_table", TestTable(id=2, name="test2", value=20.5, is_active=False)
    )
    assert setup_table.count("test_table") == 2


def test_sql(setup_table):
    setup_table.add(
        "test_table", TestTable(id=1, name="active", value=10.5, is_active=True)
    )
    setup_table.add(
        "test_table", TestTable(id=2, name="inactive", value=20.5, is_active=False)
    )

    result = setup_table.sql("SELECT * FROM test_table WHERE is_active = TRUE")
    assert len(result) == 1
    assert result[0]["is_active"] is True

    result = setup_table.sql("SELECT COUNT(*) as count FROM test_table")
    assert result[0]["count"] == 2

    # Test updating id=1 to value 100
    setup_table.sql("UPDATE test_table SET value = ? WHERE id = 1", (100,))
    assert setup_table.get("test_table", "id = 1")[0]["value"] == 100


def test_write_to_parquet(setup_table, tmp_path):
    setup_table.add(
        "test_table", TestTable(id=1, name="test1", value=10.5, is_active=True)
    )
    setup_table.add(
        "test_table", TestTable(id=2, name="test2", value=20.5, is_active=False)
    )

    file_path = tmp_path / "test_table.parquet"
    setup_table.write_to_parquet("test_table", str(file_path))

    assert file_path.exists()


def test_add_with_count(setup_table):
    # Test adding a single item
    test_item = TestTable(id=1, name="test1", value=10.5, is_active=True)
    count = len(setup_table.add("test_table", test_item))
    assert count == 1
    assert setup_table.count("test_table") == 1

    # Test adding multiple items
    test_items = [
        TestTable(id=2, name="test2", value=20.5, is_active=False),
        TestTable(id=3, name="test3", value=30.5, is_active=True),
        TestTable(id=4, name="test4", value=40.5, is_active=False),
    ]
    count = len(setup_table.add("test_table", test_items))
    assert count == 3
    assert setup_table.count("test_table") == 4

    # Test adding duplicate item (should not increase count)
    duplicate_item = TestTable(id=1, name="duplicate", value=50.5, is_active=True)
    count = len(setup_table.add("test_table", duplicate_item))
    assert count == 0
    assert setup_table.count("test_table") == 4

    # Test adding mixed new and duplicate items
    mixed_items = [
        TestTable(id=5, name="test5", value=50.5, is_active=True),
        TestTable(id=1, name="duplicate", value=60.5, is_active=False),
        TestTable(id=6, name="test6", value=70.5, is_active=True),
    ]
    count = len(setup_table.add("test_table", mixed_items))
    assert count == 2
    assert setup_table.count("test_table") == 6


def test_add_with_returning(setup_table):
    # Test adding a single item
    test_item = TestTable(id=1, name="test1", value=10.5, is_active=True)
    result = setup_table.add("test_table", [test_item])
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["name"] == "test1"
    assert result[0]["value"] == 10.5
    assert result[0]["is_active"] is True

    # Test adding multiple items
    test_items = [
        TestTable(id=2, name="test2", value=20.5, is_active=False),
        TestTable(id=3, name="test3", value=30.5, is_active=True),
    ]
    result = setup_table.add("test_table", test_items)
    assert len(result) == 2
    assert {item["id"] for item in result} == {2, 3}

    # Test adding duplicate item (should not be in the result)
    duplicate_item = TestTable(id=1, name="duplicate", value=50.5, is_active=True)
    result = setup_table.add("test_table", [duplicate_item])
    # Print table
    assert len(result) == 0

    # Test adding mixed new and duplicate items
    mixed_items = [
        TestTable(id=4, name="test4", value=40.5, is_active=False),
        TestTable(id=1, name="duplicate", value=60.5, is_active=False),
        TestTable(id=5, name="test5", value=50.5, is_active=True),
    ]
    result = setup_table.add("test_table", mixed_items)
    assert len(result) == 2
    assert {item["id"] for item in result} == {4, 5}

    # Verify total count in the table
    assert setup_table.count("test_table") == 5
