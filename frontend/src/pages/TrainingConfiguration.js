import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Play, Info } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrainingConfiguration = () => {
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [config, setConfig] = useState({
    lora_rank: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    learning_rate: 0.0002,
    num_epochs: 3,
    batch_size: 2,
    max_seq_length: 512,
    use_gpu: true,
    gradient_accumulation_steps: 4
  });
  const [targetModules, setTargetModules] = useState(['q_proj', 'v_proj', 'k_proj', 'o_proj']);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    const model = JSON.parse(localStorage.getItem('selectedModel') || 'null');
    const dataset = JSON.parse(localStorage.getItem('selectedDataset') || 'null');
    setSelectedModel(model);
    setSelectedDataset(dataset);
  }, []);

  const handleConfigChange = (field, value) => {
    setConfig({ ...config, [field]: value });
  };

  const toggleTargetModule = (module) => {
    if (targetModules.includes(module)) {
      setTargetModules(targetModules.filter(m => m !== module));
    } else {
      setTargetModules([...targetModules, module]);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedModel || !selectedDataset) {
      alert('Please select a model and dataset first');
      return;
    }

    setStarting(true);
    try {
      const trainingConfig = {
        model_id: selectedModel.id,
        dataset_id: selectedDataset.id,
        ...config,
        target_modules: targetModules
      };

      const response = await axios.post(`${API}/training/start`, trainingConfig);
      navigate(`/training/monitor/${response.data.id}`);
    } catch (error) {
      console.error('Failed to start training:', error);
      alert('Failed to start training');
      setStarting(false);
    }
  };

  const availableModules = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'gate_proj'
  ];

  return (
    <div className="space-y-6" data-testid="training-configuration-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Training Configuration</h1>
        <p className="text-slate-600 mt-1">
          Configure QLoRA parameters and hyperparameters for fine-tuning
        </p>
      </div>

      {/* Selected Model & Dataset */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <h3 className="font-semibold text-slate-900 mb-2">Selected Model</h3>
          {selectedModel ? (
            <div>
              <p className="text-sm text-slate-700">{selectedModel.name}</p>
              <p className="text-xs text-slate-500 mt-1">{selectedModel.parameters} parameters</p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No model selected</p>
          )}
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <h3 className="font-semibold text-slate-900 mb-2">Selected Dataset</h3>
          {selectedDataset ? (
            <div>
              <p className="text-sm text-slate-700">{selectedDataset.name}</p>
              <p className="text-xs text-slate-500 mt-1">{selectedDataset.rows} training examples</p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No dataset selected</p>
          )}
        </div>
      </div>

      {/* Configuration Sections */}
      <div className="space-y-6">
        {/* LoRA Parameters */}
        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Settings className="h-5 w-5 text-indigo-600" />
            <h2 className="text-lg font-semibold text-slate-900">LoRA Parameters</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                LoRA Rank (r)
              </label>
              <input
                type="number"
                data-testid="lora-rank-input"
                value={config.lora_rank}
                onChange={(e) => handleConfigChange('lora_rank', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Controls adapter capacity (4-64)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                LoRA Alpha
              </label>
              <input
                type="number"
                data-testid="lora-alpha-input"
                value={config.lora_alpha}
                onChange={(e) => handleConfigChange('lora_alpha', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Scaling factor (typically 2x rank)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                LoRA Dropout
              </label>
              <input
                type="number"
                data-testid="lora-dropout-input"
                step="0.01"
                min="0"
                max="1"
                value={config.lora_dropout}
                onChange={(e) => handleConfigChange('lora_dropout', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Regularization (0.0-0.5)</p>
            </div>
          </div>

          {/* Target Modules */}
          <div className="mt-6">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Target Modules
            </label>
            <div className="flex flex-wrap gap-2">
              {availableModules.map(module => (
                <button
                  key={module}
                  data-testid={`target-module-${module}`}
                  onClick={() => toggleTargetModule(module)}
                  className={`${
                    targetModules.includes(module)
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                  } px-3 py-1.5 rounded-md text-sm font-medium transition-colors`}
                >
                  {module}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Select which transformer layers to apply LoRA adapters
            </p>
          </div>
        </div>

        {/* Training Hyperparameters */}
        <div className="bg-white border border-slate-200 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Training Hyperparameters</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Learning Rate
              </label>
              <input
                type="number"
                data-testid="learning-rate-input"
                step="0.00001"
                value={config.learning_rate}
                onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Recommended: 1e-4 to 5e-4</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Number of Epochs
              </label>
              <input
                type="number"
                data-testid="num-epochs-input"
                min="1"
                max="10"
                value={config.num_epochs}
                onChange={(e) => handleConfigChange('num_epochs', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Full passes through dataset</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Batch Size
              </label>
              <input
                type="number"
                data-testid="batch-size-input"
                min="1"
                max="16"
                value={config.batch_size}
                onChange={(e) => handleConfigChange('batch_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Samples per training step</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Max Sequence Length
              </label>
              <input
                type="number"
                data-testid="max-seq-length-input"
                step="128"
                value={config.max_seq_length}
                onChange={(e) => handleConfigChange('max_seq_length', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Maximum token length</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Gradient Accumulation Steps
              </label>
              <input
                type="number"
                data-testid="gradient-accumulation-input"
                min="1"
                max="16"
                value={config.gradient_accumulation_steps}
                onChange={(e) => handleConfigChange('gradient_accumulation_steps', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <p className="text-xs text-slate-500 mt-1">Effective batch size multiplier</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Compute Mode
              </label>
              <select
                data-testid="compute-mode-select"
                value={config.use_gpu ? 'gpu' : 'cpu'}
                onChange={(e) => handleConfigChange('use_gpu', e.target.value === 'gpu')}
                className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="gpu">GPU (Recommended)</option>
                <option value="cpu">CPU</option>
              </select>
              <p className="text-xs text-slate-500 mt-1">Training hardware</p>
            </div>
          </div>
        </div>

        {/* Info Banner */}
        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Info className="h-5 w-5 text-indigo-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-indigo-900">
              <p className="font-medium mb-1">Training Simulation Mode</p>
              <p className="text-indigo-800">
                This is a demonstration environment. Training will be simulated with realistic progress and metrics.
                In production, these configurations would be used for actual QLoRA fine-tuning on GPU infrastructure.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <button
          onClick={() => navigate('/models')}
          className="px-6 py-2.5 border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50 transition-colors"
        >
          Back
        </button>
        <button
          data-testid="start-training-button"
          onClick={handleStartTraining}
          disabled={!selectedModel || !selectedDataset || starting}
          className="px-6 py-2.5 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
        >
          <Play className="h-5 w-5" />
          <span>{starting ? 'Starting...' : 'Start Training'}</span>
        </button>
      </div>
    </div>
  );
};

export default TrainingConfiguration;