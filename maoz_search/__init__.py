"""Public API for the MAOZ Hebrew semantic-search proof of concept.

The package contains synthetic data only.  Runtime inference is local: no profile
or query text is sent to a hosted service.
"""

from .search import SearchEngine, SearchResponse, SearchResult

__all__ = ["SearchEngine", "SearchResponse", "SearchResult"]
__version__ = "0.1.0"
