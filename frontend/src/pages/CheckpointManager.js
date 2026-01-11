import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Save, Trash2, Download, CheckCircle, HardDrive } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CheckpointManager = () => {
  const [checkpoints, setCheckpoints] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCheckpoints();
  }, []);

  const fetchCheckpoints = async () => {
    try {
      const response = await axios.get(`${API}/checkpoints`);
      setCheckpoints(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error);
      setLoading(false);
    }
  };

  const handleDelete = async (checkpointId) => {
    if (!window.confirm('Are you sure you want to delete this checkpoint?')) return;

    try {
      await axios.delete(`${API}/checkpoints/${checkpointId}`);
      fetchCheckpoints();
    } catch (error) {
      console.error('Failed to delete checkpoint:', error);
      alert('Failed to delete checkpoint');
    }
  };

  const handleDownload = (checkpoint) => {
    // Simulate download
    alert(`Downloading checkpoint: ${checkpoint.model_name}\n\nIn production, this would download the fine-tuned model weights.`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse">Loading checkpoints...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="checkpoint-manager-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Checkpoint Manager</h1>
        <p className="text-slate-600 mt-1">
          Manage saved model checkpoints from training sessions
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Total Checkpoints</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">{checkpoints.length}</p>
            </div>
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
              <Save className="h-6 w-6 text-purple-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Total Storage</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {(checkpoints.reduce((sum, cp) => sum + cp.size_mb, 0) / 1024).toFixed(2)} GB
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <HardDrive className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Latest Checkpoint</p>
              <p className="text-sm font-bold text-slate-900 mt-2">
                {checkpoints.length > 0 
                  ? new Date(checkpoints[0].created_at).toLocaleDateString()
                  : '-'
                }
              </p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Checkpoints List */}
      <div className="bg-white rounded-lg border border-slate-200">
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">Saved Checkpoints</h2>
        </div>
        <div className="p-6">
          {checkpoints.length === 0 ? (
            <div className="text-center py-12">
              <Save className="h-16 w-16 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-600 mb-2">No checkpoints saved yet</p>
              <p className="text-sm text-slate-500">
                Checkpoints will be automatically saved during training
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {checkpoints.map((checkpoint) => (
                <div
                  key={checkpoint.id}
                  data-testid={`checkpoint-${checkpoint.id}`}
                  className="border border-slate-200 rounded-lg p-4 hover:border-indigo-300 hover:bg-indigo-50/50 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                          <Save className="h-5 w-5 text-purple-600" />
                        </div>
                        <div>
                          <h3 className="font-medium text-slate-900">{checkpoint.model_name}</h3>
                          <div className="flex items-center space-x-4 mt-1 text-sm text-slate-600">
                            <span>Epoch {checkpoint.epoch}</span>
                            <span>•</span>
                            <span>Step {checkpoint.step}</span>
                            <span>•</span>
                            <span>Loss: {checkpoint.loss.toFixed(4)}</span>
                            <span>•</span>
                            <span>{checkpoint.size_mb.toFixed(2)} MB</span>
                            <span>•</span>
                            <span>{new Date(checkpoint.created_at).toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2 ml-4">
                      <button
                        data-testid={`download-checkpoint-${checkpoint.id}`}
                        onClick={() => handleDownload(checkpoint)}
                        className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
                        title="Download checkpoint"
                      >
                        <Download className="h-5 w-5" />
                      </button>
                      <button
                        data-testid={`delete-checkpoint-${checkpoint.id}`}
                        onClick={() => handleDelete(checkpoint.id)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-md transition-colors"
                        title="Delete checkpoint"
                      >
                        <Trash2 className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="text-sm text-blue-900">
          <p className="font-medium mb-1">About Checkpoints</p>
          <p className="text-blue-800">
            Checkpoints are automatically saved at the end of each training epoch. They contain the 
            fine-tuned LoRA adapter weights that can be merged with the base model or used separately 
            for inference. You can download checkpoints for deployment or resume training from them.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CheckpointManager;