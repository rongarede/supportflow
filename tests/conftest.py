from datetime import UTC, datetime

import pytest

from supportflow.domain.models import Ticket
from supportflow.workflow.service import SupportFlowService


@pytest.fixture
def duplicate_ticket() -> Ticket:
    return Ticket(
        ticket_id="ticket-duplicate-001",
        customer_id="customer-001",
        subject="I was charged twice",
        body="My order order-100 was charged twice for USD 29.00.",
        order_id="order-100",
        amount="29.00",
        currency="USD",
        created_at=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )


@pytest.fixture
def demo_service() -> SupportFlowService:
    return SupportFlowService.demo(as_of=datetime(2026, 8, 31, tzinfo=UTC))
