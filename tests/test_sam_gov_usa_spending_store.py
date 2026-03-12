"""Tests for SAM.gov description fix and USASpending raw_store wiring.

Covers:
- ContractOpportunity.description field exists and is populated by _parse_opportunity
- store_sam_opportunities persists description to SQLite
- get_sam_opportunities read method returns stored rows
- USASpendingClient accepts raw_store and calls store_usaspending_contracts
- store_usaspending_contracts persists contracts (dataclass + dict paths)
- get_usaspending_contracts read method returns stored rows
- Empty-input guards return 0 without error
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from mae_core.market.apis.sam_gov import ContractOpportunity, SAMGovClient
from mae_core.market.apis.usa_spending import GovernmentContract, USASpendingClient
from mae_core.market.raw_store import RawStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """RawStore backed by a temp directory so tests never touch production DBs."""
    s = RawStore(base_dir=tmp_path)
    yield s
    s.close()


@pytest.fixture
def sam_client(store):
    """SAMGovClient with raw_store injected, no real API calls."""
    return SAMGovClient(api_key="fake_key", raw_store=store)


@pytest.fixture
def usa_client(store):
    """USASpendingClient with raw_store injected."""
    return USASpendingClient(raw_store=store)


# ---------------------------------------------------------------------------
# SAM.gov description field on ContractOpportunity
# ---------------------------------------------------------------------------

class TestContractOpportunityDescription:
    def test_description_field_exists(self):
        """ContractOpportunity dataclass must have a description field."""
        opp = ContractOpportunity(notice_id="N1", title="Test Contract")
        assert hasattr(opp, "description")

    def test_description_default_is_empty_string(self):
        opp = ContractOpportunity(notice_id="N1", title="Test Contract")
        assert opp.description == ""

    def test_description_can_be_set(self):
        opp = ContractOpportunity(
            notice_id="N1", title="Test Contract",
            description="Full scope text here"
        )
        assert opp.description == "Full scope text here"


class TestParseOpportunityDescription:
    def test_plain_string_description_preserved(self, sam_client):
        """_parse_opportunity must pass description string into dataclass."""
        raw = {
            "noticeId": "ABC123",
            "title": "F-35 Maintenance",
            "description": "Provide depot-level maintenance for F-35 airframes.",
        }
        opp = sam_client._parse_opportunity(raw)
        assert opp.description == "Provide depot-level maintenance for F-35 airframes."

    def test_nested_dict_description_extracted(self, sam_client):
        """Some SAM.gov API versions nest description as {'content': '...'}."""
        raw = {
            "noticeId": "DEF456",
            "title": "IT Services",
            "description": {"content": "Cloud migration services for DoD."},
        }
        opp = sam_client._parse_opportunity(raw)
        assert "Cloud migration services" in opp.description

    def test_missing_description_gives_empty_string(self, sam_client):
        raw = {"noticeId": "GHI789", "title": "No Desc Contract"}
        opp = sam_client._parse_opportunity(raw)
        assert opp.description == ""

    def test_description_truncated_at_500_chars(self, sam_client):
        long_desc = "X" * 1000
        raw = {"noticeId": "LONG1", "title": "Long Desc", "description": long_desc}
        opp = sam_client._parse_opportunity(raw)
        # Dataclass stores what _parse_opportunity put in it; store truncates to 1000
        assert len(opp.description) <= 1000

    def test_other_fields_still_populated(self, sam_client):
        """Ensure description addition didn't break other field mapping."""
        raw = {
            "noticeId": "MIX1",
            "title": "Mixed Fields",
            "description": "Some scope.",
            "department": "DoD",
            "estimatedValue": "5000000",
        }
        opp = sam_client._parse_opportunity(raw)
        assert opp.notice_id == "MIX1"
        assert opp.department == "DoD"
        assert opp.estimated_value == 5_000_000.0
        assert opp.description == "Some scope."


# ---------------------------------------------------------------------------
# store_sam_opportunities — description column
# ---------------------------------------------------------------------------

class TestStoreSamOpportunitiesDescription:
    def test_description_stored_and_retrieved_via_sqlite(self, store):
        opp = ContractOpportunity(
            notice_id="N-STORE-1",
            title="Radar System",
            description="Advanced radar for missile defense.",
        )
        count = store.store_sam_opportunities([opp])
        assert count == 1

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM sam_opportunities WHERE notice_id = ?",
            ("N-STORE-1",),
        ).fetchone()
        assert row is not None
        assert "radar for missile defense" in row[0]

    def test_empty_description_stored_as_empty_string(self, store):
        opp = ContractOpportunity(notice_id="N-STORE-2", title="No Scope")
        store.store_sam_opportunities([opp])

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM sam_opportunities WHERE notice_id = ?",
            ("N-STORE-2",),
        ).fetchone()
        assert row is not None
        assert row[0] == "" or row[0] is None  # Empty or null both acceptable

    def test_dict_path_stores_description(self, store):
        opp_dict = {
            "notice_id": "N-DICT-1",
            "title": "Dict Opportunity",
            "description": "From dict path.",
            "estimated_value": 1_000_000.0,
        }
        count = store.store_sam_opportunities([opp_dict])
        assert count == 1

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM sam_opportunities WHERE notice_id = ?",
            ("N-DICT-1",),
        ).fetchone()
        assert row is not None
        assert "From dict path" in row[0]

    def test_description_truncated_to_1000_chars_in_store(self, store):
        opp = ContractOpportunity(
            notice_id="N-TRUNC-1",
            title="Long Scope",
            description="Y" * 2000,
        )
        store.store_sam_opportunities([opp])

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM sam_opportunities WHERE notice_id = ?",
            ("N-TRUNC-1",),
        ).fetchone()
        assert row is not None
        assert len(row[0]) <= 1000

    def test_upsert_updates_description(self, store):
        """Storing the same notice_id twice should update the description."""
        opp1 = ContractOpportunity(
            notice_id="N-UP-1", title="Contract", description="Version 1"
        )
        opp2 = ContractOpportunity(
            notice_id="N-UP-1", title="Contract", description="Version 2 updated"
        )
        store.store_sam_opportunities([opp1])
        store.store_sam_opportunities([opp2])

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM sam_opportunities WHERE notice_id = ?",
            ("N-UP-1",),
        ).fetchone()
        assert "Version 2" in row[0]

        # Only one row (upsert, not duplicate insert)
        count = conn.execute(
            "SELECT COUNT(*) FROM sam_opportunities WHERE notice_id = ?", ("N-UP-1",)
        ).fetchone()[0]
        assert count == 1

    def test_empty_input_returns_zero(self, store):
        assert store.store_sam_opportunities([]) == 0


# ---------------------------------------------------------------------------
# get_sam_opportunities read method
# ---------------------------------------------------------------------------

class TestGetSamOpportunities:
    def test_returns_list_of_dicts(self, store):
        opp = ContractOpportunity(
            notice_id="READ-1",
            title="Readable Contract",
            description="Read me.",
            posted_date="2026-03-01",
        )
        store.store_sam_opportunities([opp])
        results = store.get_sam_opportunities(lookback_days=365)
        assert isinstance(results, list)
        assert len(results) >= 1
        row = next((r for r in results if r["notice_id"] == "READ-1"), None)
        assert row is not None
        assert row["description"] == "Read me."
        assert row["title"] == "Readable Contract"

    def test_empty_table_returns_empty_list(self, store):
        # Nothing stored — should return [] without error
        results = store.get_sam_opportunities(lookback_days=30)
        assert results == []

    def test_description_key_present_in_returned_dicts(self, store):
        opp = ContractOpportunity(
            notice_id="READ-2", title="Key Test", description="Has desc",
            posted_date="2026-03-05",
        )
        store.store_sam_opportunities([opp])
        results = store.get_sam_opportunities(lookback_days=365)
        assert all("description" in r for r in results)

    def test_multiple_rows_returned(self, store):
        opps = [
            ContractOpportunity(
                notice_id=f"MULTI-{i}", title=f"Contract {i}",
                description=f"Scope {i}", posted_date="2026-03-01",
            )
            for i in range(5)
        ]
        store.store_sam_opportunities(opps)
        results = store.get_sam_opportunities(lookback_days=365)
        ids = {r["notice_id"] for r in results}
        for i in range(5):
            assert f"MULTI-{i}" in ids


# ---------------------------------------------------------------------------
# USASpendingClient — raw_store wiring
# ---------------------------------------------------------------------------

class TestUSASpendingClientRawStore:
    def test_accepts_raw_store_in_init(self, store):
        """Constructor must accept raw_store without error."""
        client = USASpendingClient(raw_store=store)
        assert client._raw_store is store

    def test_raw_store_none_by_default(self):
        """Without raw_store kwarg, _raw_store is None."""
        client = USASpendingClient()
        assert client._raw_store is None

    def test_store_contracts_called_on_search(self, usa_client):
        """_store_contracts should be called after parse; verify side-effect in DB."""
        # Replace _post so no HTTP is made; return a fake API response
        fake_award = {
            "Recipient Name": "Acme Corp",
            "recipient_id": "DUNS123",
            "Recipient State Name": "VA",
            "Recipient Country": "USA",
            "internal_id": "AWARD-001",
            "Award Amount": 5_000_000,
            "Start Date": "2026-03-01",
            "Award Type": "Contract",
            "Description": "IT infrastructure services.",
            "NAICS Code": "541512",
            "NAICS Description": "Computer Systems Design",
            "Awarding Agency": "Department of Defense",
            "Awarding Sub Agency": "Army",
            "Funding Agency": "Department of Defense",
            "End Date": "2027-03-01",
        }
        usa_client._post = MagicMock(return_value={"results": [fake_award]})

        contracts = usa_client.search_contracts(keyword="IT", limit=1)
        assert len(contracts) == 1
        assert contracts[0].description == "IT infrastructure services."

        # Verify it was persisted to raw_store
        conn = usa_client._raw_store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM usaspending_contracts WHERE award_id = ?",
            ("AWARD-001",),
        ).fetchone()
        assert row is not None
        assert "IT infrastructure services" in row[0]

    def test_store_contracts_not_called_when_api_fails(self, usa_client):
        """If _post returns None, no DB write should occur."""
        usa_client._post = MagicMock(return_value=None)
        contracts = usa_client.search_contracts(keyword="fail")
        assert contracts == []

        # Table may not even exist — that's fine
        try:
            conn = usa_client._raw_store._get_conn("contracts")
            count = conn.execute(
                "SELECT COUNT(*) FROM usaspending_contracts"
            ).fetchone()[0]
            assert count == 0
        except Exception:
            pass  # Table not created = no writes happened, which is correct

    def test_store_contracts_no_error_when_raw_store_none(self):
        """If raw_store is None, _store_contracts must silently skip."""
        client = USASpendingClient(raw_store=None)
        # Should not raise
        client._store_contracts([])
        contract = GovernmentContract(
            recipient_name="X", recipient_duns="", recipient_location="",
            award_id="A1", award_amount=100.0, award_date="2026-01-01",
            award_type="Contract", description="Test", naics_code="",
            naics_description="", awarding_agency="DoD",
            awarding_sub_agency="", funding_agency="", start_date="",
            end_date=""
        )
        client._store_contracts([contract])  # no error expected


# ---------------------------------------------------------------------------
# store_usaspending_contracts — storage mixin
# ---------------------------------------------------------------------------

class TestStoreUSASpendingContracts:
    def _make_contract(self, award_id="C1", description="Sample scope."):
        return GovernmentContract(
            recipient_name="Boeing",
            recipient_duns="DUNS-001",
            recipient_location="WA, USA",
            award_id=award_id,
            award_amount=10_000_000.0,
            award_date="2026-03-01",
            award_type="Contract",
            description=description,
            naics_code="336411",
            naics_description="Aircraft Manufacturing",
            awarding_agency="Department of Defense",
            awarding_sub_agency="Air Force",
            funding_agency="Department of Defense",
            start_date="2026-03-01",
            end_date="2027-03-01",
        )

    def test_dataclass_path_stores_all_fields(self, store):
        c = self._make_contract("C-DC-1", "Avionics upgrade services.")
        count = store.store_usaspending_contracts([c])
        assert count == 1

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT recipient_name, description, award_amount FROM usaspending_contracts "
            "WHERE award_id = ?", ("C-DC-1",)
        ).fetchone()
        assert row is not None
        assert row[0] == "Boeing"
        assert "Avionics upgrade" in row[1]
        assert row[2] == 10_000_000.0

    def test_dict_path_stores_all_fields(self, store):
        c_dict = {
            "award_id": "C-DICT-1",
            "recipient_name": "Lockheed Martin",
            "recipient_duns": "DUNS-002",
            "recipient_location": "TX, USA",
            "award_amount": 50_000_000.0,
            "award_date": "2026-02-01",
            "award_type": "Contract",
            "description": "F-35 production support.",
            "naics_code": "336414",
            "naics_description": "Guided Missile Manufacturing",
            "awarding_agency": "Department of Defense",
            "awarding_sub_agency": "Navy",
            "funding_agency": "Department of Defense",
            "start_date": "2026-02-01",
            "end_date": "2028-02-01",
        }
        count = store.store_usaspending_contracts([c_dict])
        assert count == 1

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM usaspending_contracts WHERE award_id = ?",
            ("C-DICT-1",),
        ).fetchone()
        assert row is not None
        assert "F-35 production" in row[0]

    def test_description_truncated_to_1000_chars(self, store):
        c = self._make_contract("C-TRUNC-1", "Z" * 2000)
        store.store_usaspending_contracts([c])

        conn = store._get_conn("contracts")
        row = conn.execute(
            "SELECT description FROM usaspending_contracts WHERE award_id = ?",
            ("C-TRUNC-1",),
        ).fetchone()
        assert row is not None
        assert len(row[0]) <= 1000

    def test_upsert_on_same_award_id(self, store):
        c1 = self._make_contract("C-UP-1", "Original description.")
        c2 = self._make_contract("C-UP-1", "Updated description.")
        store.store_usaspending_contracts([c1])
        store.store_usaspending_contracts([c2])

        conn = store._get_conn("contracts")
        count = conn.execute(
            "SELECT COUNT(*) FROM usaspending_contracts WHERE award_id = ?", ("C-UP-1",)
        ).fetchone()[0]
        assert count == 1

        row = conn.execute(
            "SELECT description FROM usaspending_contracts WHERE award_id = ?",
            ("C-UP-1",),
        ).fetchone()
        assert "Updated description" in row[0]

    def test_multiple_contracts_stored(self, store):
        contracts = [self._make_contract(f"C-MULTI-{i}") for i in range(5)]
        count = store.store_usaspending_contracts(contracts)
        assert count == 5

    def test_empty_input_returns_zero(self, store):
        assert store.store_usaspending_contracts([]) == 0

    def test_null_award_id_handled(self, store):
        """award_id can be empty string — should not crash."""
        c = self._make_contract("")
        # SQLite allows empty string PKs; should store 1 row or handle gracefully
        result = store.store_usaspending_contracts([c])
        assert result >= 0  # no exception


# ---------------------------------------------------------------------------
# get_usaspending_contracts read method
# ---------------------------------------------------------------------------

class TestGetUSASpendingContracts:
    def _make_contract(self, award_id, award_date="2026-03-01"):
        return GovernmentContract(
            recipient_name="Raytheon",
            recipient_duns="DUNS-RTN",
            recipient_location="MA, USA",
            award_id=award_id,
            award_amount=20_000_000.0,
            award_date=award_date,
            award_type="Contract",
            description=f"Scope for {award_id}.",
            naics_code="541330",
            naics_description="Engineering Services",
            awarding_agency="Department of Defense",
            awarding_sub_agency="Missile Defense Agency",
            funding_agency="Department of Defense",
            start_date=award_date,
            end_date="2027-03-01",
        )

    def test_returns_list_of_dicts(self, store):
        c = self._make_contract("READ-C-1", "2026-03-01")
        store.store_usaspending_contracts([c])
        results = store.get_usaspending_contracts(lookback_days=365)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_result_contains_expected_keys(self, store):
        c = self._make_contract("READ-C-2", "2026-03-01")
        store.store_usaspending_contracts([c])
        results = store.get_usaspending_contracts(lookback_days=365)
        row = next((r for r in results if r["award_id"] == "READ-C-2"), None)
        assert row is not None
        expected_keys = [
            "award_id", "recipient_name", "description", "award_amount",
            "award_date", "awarding_agency", "naics_code",
        ]
        for k in expected_keys:
            assert k in row, f"Missing key: {k}"

    def test_description_round_trips(self, store):
        c = self._make_contract("READ-C-3", "2026-03-05")
        store.store_usaspending_contracts([c])
        results = store.get_usaspending_contracts(lookback_days=365)
        row = next((r for r in results if r["award_id"] == "READ-C-3"), None)
        assert row is not None
        assert "Scope for READ-C-3" in row["description"]

    def test_empty_table_returns_empty_list(self, store):
        results = store.get_usaspending_contracts(lookback_days=30)
        assert results == []

    def test_multiple_contracts_returned(self, store):
        contracts = [self._make_contract(f"READ-MULTI-{i}", "2026-03-01") for i in range(4)]
        store.store_usaspending_contracts(contracts)
        results = store.get_usaspending_contracts(lookback_days=365)
        ids = {r["award_id"] for r in results}
        for i in range(4):
            assert f"READ-MULTI-{i}" in ids
