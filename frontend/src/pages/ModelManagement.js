import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Brain, 
  Download, 
  Trash2, 
  Play, 
  Settings, 
  CheckCircle, 
  AlertCircle,
  Clock,
  HardDrive,
  Zap,
  Plus,
  Search,
  Filter
} from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [showInferenceDialog, setShowInferenceDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [mergeForm, setMergeForm] = useState({
    base_model_id: '',
    adapter_path: '',
    output_name: '',
    use_4bit: false
  });
  const [inferenceForm, setInferenceForm] = useState({
    model_path: '',
    prompt: '',
    max_tokens: 200,
    temperature: 0.7,
    top_p: 0.9,
    do_sample: true
  });
  const [inferenceResult, setInferenceResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Load data
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [modelsRes, jobsRes] = await Promise.all([
        axios.get(`${API}/models/merged`),
        axios.get(`${API}/training/jobs`)
      ]);
      
      setModels(modelsRes.data.models || []);
      setJobs(jobsRes.data || []);
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handle merge model
  const handleMergeModel = async () => {
    if (!mergeForm.base_model_id || !mergeForm.adapter_path || !mergeForm.output_name) {
      alert('Please fill all required fields');
      return;
    }

    try {
      setIsProcessing(true);
      const formData = new FormData();
      formData.append('base_model_id', mergeForm.base_model_id);
      formData.append('adapter_path', mergeForm.adapter_path);
      formData.append('output_name', mergeForm.output_name);
      formData.append('use_4bit', mergeForm.use_4bit);

      const response = await axios.post(`${API}/models/merge`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      alert('Model merged successfully!');
      setShowMergeDialog(false);
      setMergeForm({
        base_model_id: '',
        adapter_path: '',
        output_name: '',
        use_4bit: false
      });
      loadData();
    } catch (error) {
      console.error('Merge failed:', error);
      alert(`Failed to merge model: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle inference
  const handleInference = async () => {
    if (!inferenceForm.model_path || !inferenceForm.prompt) {
      alert('Please provide model path and prompt');
      return;
    }

    try {
      setIsProcessing(true);
      const formData = new FormData();
      formData.append('model_path', inferenceForm.model_path);
      formData.append('prompt', inferenceForm.prompt);
      formData.append('max_tokens', inferenceForm.max_tokens);
      formData.append('temperature', inferenceForm.temperature);
      formData.append('top_p', inferenceForm.top_p);
      formData.append('do_sample', inferenceForm.do_sample);

      const response = await axios.post(`${API}/models/inference`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setInferenceResult(response.data);
    } catch (error) {
      console.error('Inference failed:', error);
      alert(`Failed to run inference: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle delete model
  const handleDeleteModel = async (modelName) => {
    if (!confirm(`Are you sure you want to delete model "${modelName}"?`)) {
      return;
    }

    try {
      await axios.delete(`${API}/models/merged/${modelName}`);
      alert('Model deleted successfully!');
      loadData();
    } catch (error) {
      console.error('Delete failed:', error);
      alert(`Failed to delete model: ${error.response?.data?.detail || error.message}`);
    }
  };

  // Filter models
  const filteredModels = models.filter(model => 
    model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    model.metadata.base_model_id?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Filter completed jobs for merge dialog
  const completedJobs = jobs.filter(job => job.status === 'completed');

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 flex items-center">
            <Brain className="h-8 w-8 mr-3 text-blue-600" />
            TechnoFriendR Model Management
          </h1>
          <p className="text-slate-600 mt-1">Merge, test, and manage your AI fine-tuned models</p>
        </div>
        <button
          onClick={() => setShowMergeDialog(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center space-x-2"
        >
          <Plus className="h-5 w-5" />
          <span>Merge Model</span>
        </button>
      </div>

      {/* Search and Filter */}
      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 mb-6">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search models..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          <button className="px-4 py-2 border border-slate-300 rounded-md hover:bg-slate-50 flex items-center space-x-2">
            <Filter className="h-4 w-4" />
            <span>Filter</span>
          </button>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredModels.map((model) => (
          <div key={model.name} className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-blue-600 mr-3" />
                <div>
                  <h3 className="font-semibold text-slate-900">{model.name}</h3>
                  <p className="text-sm text-slate-500 truncate max-w-[200px]">
                    {model.metadata.base_model_id}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-1">
                <CheckCircle className="h-4 w-4 text-green-500" />
              </div>
            </div>

            <div className="space-y-3 mb-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Size:</span>
                <span className="font-medium">{model.size_mb} MB</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Created:</span>
                <span className="font-medium">
                  {new Date(model.created_at).toLocaleDateString()}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Base Model:</span>
                <span className="font-medium truncate max-w-[120px]" title={model.metadata.base_model_id}>
                  {model.metadata.base_model_id?.split('/')?.pop() || 'Unknown'}
                </span>
              </div>
              {model.metadata.use_4bit && (
                <div className="flex items-center text-sm text-blue-600">
                  <Zap className="h-3 w-3 mr-1" />
                  <span>4-bit Quantized</span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  setInferenceForm({ ...inferenceForm, model_path: model.path });
                  setShowInferenceDialog(true);
                }}
                className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center justify-center space-x-1"
              >
                <Play className="h-4 w-4" />
                <span>Test</span>
              </button>
              <button
                onClick={() => handleDeleteModel(model.name)}
                className="px-3 py-2 border border-red-300 text-red-600 rounded-md hover:bg-red-50 transition-colors"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}

        {/* Empty State */}
        {filteredModels.length === 0 && (
          <div className="col-span-full text-center py-12">
            <Brain className="h-16 w-16 text-slate-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-slate-900 mb-2">No models found</h3>
            <p className="text-slate-500 mb-4">Merge your fine-tuned adapters to create models</p>
            <button
              onClick={() => setShowMergeDialog(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Merge Your First Model
            </button>
          </div>
        )}
      </div>

      {/* Merge Dialog */}
      {showMergeDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold text-slate-900 mb-4">Merge Model</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Base Model ID
                </label>
                <input
                  type="text"
                  placeholder="e.g., meta-llama/Llama-2-7b-hf"
                  value={mergeForm.base_model_id}
                  onChange={(e) => setMergeForm({ ...mergeForm, base_model_id: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Adapter Path
                </label>
                <select
                  value={mergeForm.adapter_path}
                  onChange={(e) => setMergeForm({ ...mergeForm, adapter_path: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Select completed training job...</option>
                  {completedJobs.map((job) => (
                    <option key={job.id} value={job.model_path}>
                      {job.id} - {job.config?.model_id || 'Unknown Model'}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Output Model Name
                </label>
                <input
                  type="text"
                  placeholder="e.g., my-finetuned-model"
                  value={mergeForm.output_name}
                  onChange={(e) => setMergeForm({ ...mergeForm, output_name: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="use_4bit"
                  checked={mergeForm.use_4bit}
                  onChange={(e) => setMergeForm({ ...mergeForm, use_4bit: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
                />
                <label htmlFor="use_4bit" className="ml-2 text-sm text-slate-700">
                  Use 4-bit quantization (saves memory)
                </label>
              </div>
            </div>

            <div className="flex items-center justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowMergeDialog(false)}
                className="px-4 py-2 border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={handleMergeModel}
                disabled={isProcessing}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {isProcessing ? 'Merging...' : 'Merge Model'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Inference Dialog */}
      {showInferenceDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold text-slate-900 mb-4">Test Model Inference</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Model Path
                </label>
                <input
                  type="text"
                  value={inferenceForm.model_path}
                  readOnly
                  className="w-full px-3 py-2 border border-slate-300 rounded-md bg-slate-50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Prompt
                </label>
                <textarea
                  placeholder="Enter your prompt here..."
                  value={inferenceForm.prompt}
                  onChange={(e) => setInferenceForm({ ...inferenceForm, prompt: e.target.value })}
                  rows={3}
                  className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    value={inferenceForm.max_tokens}
                    onChange={(e) => setInferenceForm({ ...inferenceForm, max_tokens: parseInt(e.target.value) })}
                    min="1"
                    max="1024"
                    className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Temperature
                  </label>
                  <input
                    type="number"
                    value={inferenceForm.temperature}
                    onChange={(e) => setInferenceForm({ ...inferenceForm, temperature: parseFloat(e.target.value) })}
                    min="0.1"
                    max="2.0"
                    step="0.1"
                    className="w-full px-3 py-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="do_sample"
                  checked={inferenceForm.do_sample}
                  onChange={(e) => setInferenceForm({ ...inferenceForm, do_sample: e.target.checked })}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
                />
                <label htmlFor="do_sample" className="ml-2 text-sm text-slate-700">
                  Use sampling (more creative responses)
                </label>
              </div>
            </div>

            {/* Inference Result */}
            {inferenceResult && (
              <div className="mt-6 p-4 bg-slate-50 rounded-lg">
                <h3 className="font-medium text-slate-900 mb-2">Response:</h3>
                <p className="text-slate-700 whitespace-pre-wrap">{inferenceResult.response}</p>
                <div className="mt-3 text-xs text-slate-500">
                  Parameters: max_tokens={inferenceResult.parameters.max_tokens}, 
                  temperature={inferenceResult.parameters.temperature}, 
                  do_sample={inferenceResult.parameters.do_sample}
                </div>
              </div>
            )}

            <div className="flex items-center justify-end space-x-3 mt-6">
              <button
                onClick={() => {
                  setShowInferenceDialog(false);
                  setInferenceResult(null);
                  setInferenceForm({
                    model_path: '',
                    prompt: '',
                    max_tokens: 200,
                    temperature: 0.7,
                    top_p: 0.9,
                    do_sample: true
                  });
                }}
                className="px-4 py-2 border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50"
              >
                Close
              </button>
              <button
                onClick={handleInference}
                disabled={isProcessing}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {isProcessing ? 'Generating...' : 'Generate Response'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelManagement;
