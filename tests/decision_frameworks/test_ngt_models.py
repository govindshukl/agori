import pytest

from agori.decision_frameworks.ngt.models import NGTExpert, NGTResponse


def test_ngt_expert_creation():
    """Test NGTExpert creation and validation."""
    expert = NGTExpert(
        role="Technical Architect",
        description="Test description",
        specialty="Test specialty",
    )

    assert expert.role == "Technical Architect"
    assert expert.description == "Test description"
    assert expert.specialty == "Test specialty"
    assert isinstance(expert.metadata, dict)


def test_ngt_expert_equality():
    """Test equality comparison of NGTExperts."""
    expert1 = NGTExpert(
        role="Technical Architect",
        description="System design expert",
        specialty="Enterprise Architecture",
    )

    expert2 = NGTExpert(
        role="Technical Architect",
        description="System design expert",
        specialty="Enterprise Architecture",
    )

    expert3 = NGTExpert(
        role="Project Manager",
        description="Project management expert",
        specialty="Agile",
    )

    assert expert1 == expert2  # Same attributes should be equal
    assert expert1 != expert3  # Different attributes should not be equal
    assert hash(expert1) == hash(expert2)  # Same attributes should have same hash


def test_ngt_response_validation():
    """Test NGTResponse validation."""
    expert = NGTExpert(
        role="Technical Architect",
        description="System design expert",
        specialty="Enterprise Architecture",
    )

    # Test valid creation
    response = NGTResponse(expert=expert, ideas="Test ideas", metrics={"count": 1})
    assert isinstance(response, NGTResponse)

    # Test invalid expert type
    with pytest.raises(TypeError):
        NGTResponse(
            expert="not an expert", ideas="test", metrics={}  # Should raise TypeError
        )

    # Test invalid metrics type
    with pytest.raises(TypeError):
        NGTResponse(
            expert=expert, ideas="test", metrics="not a dict"  # Should raise TypeError
        )

    # Test invalid ideas type
    with pytest.raises(TypeError):
        NGTResponse(
            expert=expert, ideas=["not a string"], metrics={}  # Should raise TypeError
        )
