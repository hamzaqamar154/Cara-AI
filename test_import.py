"""Test script to verify all imports work correctly."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    print("Testing imports...")
    from src.config import settings, ensure_directories
    print("✓ Config imported")
    
    from src.embedder import EmbeddingService
    print("✓ Embedder imported")
    
    from src.retriever import VectorStore
    print("✓ Retriever imported")
    
    from src.llm import LLMService
    print("✓ LLM imported")
    
    from src.data_processing import process_pdf
    print("✓ Data processing imported")
    
    print("\nAll imports successful!")
    sys.exit(0)
except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

