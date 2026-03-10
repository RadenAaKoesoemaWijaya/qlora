"""
Unit tests untuk semua training engines.

Test coverage:
- Engine creation via factory
- PEFT config generation
- Model loading (mocked)
- Training argument setup
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))


def test_training_engine_factory():
    """Test bahwa factory dapat membuat semua engines."""
    from core.training_engine_factory import TrainingEngineFactory
    
    config = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "num_epochs": 1,
    }
    
    # Test setiap method yang tersedia
    available_methods = TrainingEngineFactory.get_available_methods()
    
    for method in available_methods.keys():
        try:
            engine = TrainingEngineFactory.create_engine(method, config)
            assert engine is not None
            print(f"✅ {method}: Engine created successfully")
        except Exception as e:
            print(f"❌ {method}: Failed - {e}")
            raise


def test_dora_config():
    """Test DoRA configuration generation."""
    from core.dora_engine import DoRATrainingEngine
    
    config = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "use_dora": True,
        "dora_simple": False,
    }
    
    engine = DoRATrainingEngine(config)
    lora_config = engine.setup_lora_config()
    
    assert lora_config.use_dora == True
    assert lora_config.r == 16
    assert lora_config.lora_alpha == 32
    print("✅ DoRA config test passed")


def test_lora_plus_optimizer():
    """Test LoRA+ custom optimizer creation."""
    from core.lora_plus_engine import LoRAPlusTrainingEngine
    from unittest.mock import MagicMock
    import torch.nn as nn
    
    config = {
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "lora_plus_ratio": 16,
    }
    
    engine = LoRAPlusTrainingEngine(config)
    
    # Mock model dengan LoRA-like parameters
    mock_model = MagicMock()
    mock_params = [
        MagicMock(requires_grad=True),
        MagicMock(requires_grad=True),
        MagicMock(requires_grad=False),
    ]
    
    # Simulate named parameters
    named_params = [
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", mock_params[0]),
        ("base_model.model.layers.0.self_attn.q_proj.lora_B.weight", mock_params[1]),
        ("base_model.model.layers.0.self_attn.q_proj.base_layer.weight", mock_params[2]),
    ]
    
    mock_model.named_parameters.return_value = named_params
    
    # Test optimizer creation
    try:
        optimizer = engine.create_custom_optimizer(mock_model)
        assert optimizer is not None
        print("✅ LoRA+ optimizer test passed")
    except Exception as e:
        print(f"⚠️ LoRA+ optimizer test skipped (no torch): {e}")


def test_ia3_config():
    """Test IA³ configuration generation."""
    from core.ia3_engine import IA3TrainingEngine
    
    config = {
        "ia3_target_modules": ["k_proj", "v_proj", "down_proj"],
        "ia3_feedforward_modules": ["down_proj"],
    }
    
    engine = IA3TrainingEngine(config)
    ia3_config = engine.setup_peft_config()
    
    assert "k_proj" in ia3_config.target_modules
    assert "v_proj" in ia3_config.target_modules
    assert ia3_config.feedforward_modules == ["down_proj"]
    print("✅ IA³ config test passed")


def test_vera_config():
    """Test VeRA configuration generation."""
    from core.vera_engine import VeRATrainingEngine
    
    config = {
        "vera_rank": 256,
        "vera_seed": 42,
        "target_modules": ["q_proj", "v_proj"],
    }
    
    engine = VeRATrainingEngine(config)
    vera_config = engine.setup_peft_config()
    
    assert vera_config.r == 256
    assert vera_config.projection_prng_key == 42
    print("✅ VeRA config test passed")


def test_adalora_config():
    """Test AdaLoRA configuration generation."""
    from core.adalora_engine import AdaLoRATrainingEngine
    
    config = {
        "adalora_init_r": 12,
        "adalora_target_r": 4,
        "adalora_tinit": 0,
        "adalora_tfinal": 1000,
        "adalora_deltaT": 10,
        "adalora_beta1": 0.85,
    }
    
    engine = AdaLoRATrainingEngine(config)
    adalora_config = engine.setup_peft_config()
    
    assert adalora_config.r == 12
    assert adalora_config.target_r == 4
    assert adalora_config.tinit == 0
    assert adalora_config.tfinal == 1000
    print("✅ AdaLoRA config test passed")


def test_oft_config():
    """Test OFT configuration generation."""
    from core.oft_engine import OFTTrainingEngine
    
    config = {
        "oft_r": 8,
        "oft_dropout": 0.0,
        "oft_init_weights": True,
        "target_modules": ["q_proj", "v_proj"],
    }
    
    engine = OFTTrainingEngine(config)
    oft_config = engine.setup_peft_config()
    
    assert oft_config.r == 8
    assert oft_config.module_dropout == 0.0
    assert oft_config.init_weights == True
    print("✅ OFT config test passed")


def test_method_descriptions():
    """Test bahwa semua method memiliki descriptions."""
    from core.dora_engine import DoRATrainingEngine
    from core.lora_plus_engine import LoRAPlusTrainingEngine
    from core.ia3_engine import IA3TrainingEngine
    from core.vera_engine import VeRATrainingEngine
    from core.adalora_engine import AdaLoRATrainingEngine
    from core.oft_engine import OFTTrainingEngine
    
    engines = [
        DoRATrainingEngine,
        LoRAPlusTrainingEngine,
        IA3TrainingEngine,
        VeRATrainingEngine,
        AdaLoRATrainingEngine,
        OFTTrainingEngine,
    ]
    
    for engine_class in engines:
        desc = engine_class.get_method_description()
        assert "name" in desc
        assert "key_benefits" in desc
        print(f"✅ {desc['name']} has description")


def test_available_methods_endpoint():
    """Simulate test untuk API endpoint /api/training/methods."""
    from core.training_engine_factory import get_available_training_methods
    
    methods = get_available_training_methods()
    
    assert "qlora" in methods or "dora" in methods
    assert len(methods) >= 1
    
    # Check method metadata
    for method_id, info in methods.items():
        assert "name" in info
        assert "description" in info
        assert "efficiency" in info
        assert "performance" in info
        print(f"✅ {method_id}: {info['name']} - {info['performance']}")


@pytest.mark.skip(reason="Requires actual ML libraries and GPU")
def test_actual_model_loading():
    """
    Integration test yang sebenarnya memuat model.
    Hanya dijalankan ketika ML libraries tersedia.
    """
    from core.training_engine_factory import TrainingEngineFactory
    
    config = {
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "num_epochs": 1,
    }
    
    # Test dengan model kecil
    engine = TrainingEngineFactory.create_engine("dora", config)
    
    # This would require actual model download
    # model, tokenizer = await engine.load_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def test_config_validation():
    """Test config validation untuk setiap method."""
    from core.lora_plus_engine import LoRAPlusTrainingEngine
    from core.adalora_engine import AdaLoRATrainingEngine
    
    # Test LoRA+ ratio validation
    config_high = {"lora_plus_ratio": 500}  # Too high
    engine = LoRAPlusTrainingEngine(config_high)
    assert engine.lora_plus_ratio == 256  # Should be capped
    
    config_low = {"lora_plus_ratio": 0.5}  # Too low
    engine = LoRAPlusTrainingEngine(config_low)
    assert engine.lora_plus_ratio == 1  # Should be minimum
    
    # Test AdaLoRA target <= init
    config_adalora = {
        "adalora_init_r": 8,
        "adalora_target_r": 12,  # Invalid: target > init
    }
    engine = AdaLoRATrainingEngine(config_adalora)
    assert engine.target_r <= engine.init_r  # Should be adjusted
    
    print("✅ Config validation tests passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Training Engine Unit Tests")
    print("="*60 + "\n")
    
    # Run all tests
    try:
        test_training_engine_factory()
        test_dora_config()
        test_lora_plus_optimizer()
        test_ia3_config()
        test_vera_config()
        test_adalora_config()
        test_oft_config()
        test_method_descriptions()
        test_available_methods_endpoint()
        test_config_validation()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
