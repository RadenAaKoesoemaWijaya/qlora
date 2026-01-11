import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Brain, CheckCircle, Info } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ModelSelection = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API}/models`);
      setModels(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch models:', error);
      setLoading(false);
    }
  };

  const handleSelectModel = (model) => {
    setSelectedModel(model.id);
    localStorage.setItem('selectedModel', JSON.stringify(model));
  };

  const handleContinue = () => {
    if (selectedModel) {
      navigate('/training/configure');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="model-selection-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Select Base Model</h1>
        <p className="text-slate-600 mt-1">
          Choose a pre-trained LLM to fine-tune for medical decision support
        </p>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Info className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900">
            <p className="font-medium mb-1">About Model Selection</p>
            <p className="text-blue-800">
              These models will be fine-tuned using QLoRA (4-bit quantization with LoRA adapters) 
              to efficiently adapt them to medical decision support tasks while minimizing memory requirements.
            </p>
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((model) => (
          <div
            key={model.id}
            data-testid={`model-card-${model.id}`}
            onClick={() => handleSelectModel(model)}
            className={`${
              selectedModel === model.id
                ? 'border-indigo-600 bg-indigo-50 shadow-md'
                : 'border-slate-200 hover:border-indigo-300'
            } relative bg-white border-2 rounded-lg p-6 cursor-pointer transition-all card-hover`}
          >
            {selectedModel === model.id && (
              <div className="absolute top-4 right-4">
                <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center">
                  <CheckCircle className="h-5 w-5 text-white" />
                </div>
              </div>
            )}

            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <Brain className="h-6 w-6 text-indigo-600" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-slate-900">{model.name}</h3>
                <p className="text-sm text-slate-600 mt-1">{model.description}</p>
                
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <p className="text-xs text-slate-500">Provider</p>
                    <p className="text-sm font-medium text-slate-900">{model.provider}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Parameters</p>
                    <p className="text-sm font-medium text-slate-900">{model.parameters}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Type</p>
                    <p className="text-sm font-medium text-slate-900">{model.type}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Size</p>
                    <p className="text-sm font-medium text-slate-900">{model.size}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <button
          onClick={() => navigate('/datasets')}
          className="px-6 py-2.5 border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50 transition-colors"
        >
          Back to Datasets
        </button>
        <button
          data-testid="continue-to-training-button"
          onClick={handleContinue}
          disabled={!selectedModel}
          className="px-6 py-2.5 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Continue to Training Configuration
        </button>
      </div>
    </div>
  );
};

export default ModelSelection;