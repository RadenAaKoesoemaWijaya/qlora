import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart3, TrendingUp, Target, Award, Activity } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Evaluation = () => {
  const [evaluations, setEvaluations] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [evaluating, setEvaluating] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [evalsRes, modelsRes, datasetsRes] = await Promise.all([
        axios.get(`${API}/evaluations`),
        axios.get(`${API}/models`),
        axios.get(`${API}/datasets`)
      ]);
      setEvaluations(evalsRes.data);
      setModels(modelsRes.data);
      setDatasets(datasetsRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch data:', error);
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!selectedModel || !selectedDataset) {
      alert('Please select both model and dataset');
      return;
    }

    setEvaluating(true);
    try {
      await axios.post(`${API}/evaluate`, null, {
        params: {
          model_id: selectedModel,
          dataset_id: selectedDataset
        }
      });
      await fetchData();
    } catch (error) {
      console.error('Evaluation failed:', error);
      alert('Evaluation failed');
    } finally {
      setEvaluating(false);
    }
  };

  const latestEval = evaluations[0];

  const radarData = latestEval ? [
    { metric: 'Accuracy', value: latestEval.accuracy * 100 },
    { metric: 'Precision', value: latestEval.precision * 100 },
    { metric: 'Recall', value: latestEval.recall * 100 },
    { metric: 'F1-Score', value: latestEval.f1_score * 100 },
  ] : [];

  const barData = evaluations.slice(0, 5).reverse().map((eval, idx) => ({
    name: `Eval ${idx + 1}`,
    accuracy: eval.accuracy * 100,
    f1Score: eval.f1_score * 100
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse">Loading evaluation data...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="evaluation-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Model Evaluation</h1>
        <p className="text-slate-600 mt-1">
          Evaluate fine-tuned models on test datasets
        </p>
      </div>

      {/* Evaluation Form */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Run Evaluation</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Select Model
            </label>
            <select
              data-testid="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Choose a model...</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>{model.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Select Test Dataset
            </label>
            <select
              data-testid="dataset-select"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Choose a dataset...</option>
              {datasets.map(dataset => (
                <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              data-testid="evaluate-button"
              onClick={handleEvaluate}
              disabled={!selectedModel || !selectedDataset || evaluating}
              className="w-full px-4 py-2 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
            >
              {evaluating ? (
                <>
                  <Activity className="h-5 w-5 animate-spin" />
                  <span>Evaluating...</span>
                </>
              ) : (
                <>
                  <BarChart3 className="h-5 w-5" />
                  <span>Evaluate</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Latest Results */}
      {latestEval && (
        <>
          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Accuracy</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">
                    {(latestEval.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <Target className="h-6 w-6 text-green-600" />
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Precision</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">
                    {(latestEval.precision * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Recall</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">
                    {(latestEval.recall * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <BarChart3 className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">F1 Score</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">
                    {(latestEval.f1_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
                  <Award className="h-6 w-6 text-indigo-600" />
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Perplexity</p>
                  <p className="text-3xl font-bold text-slate-900 mt-2">
                    {latestEval.perplexity.toFixed(2)}
                  </p>
                </div>
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                  <Activity className="h-6 w-6 text-orange-600" />
                </div>
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Radar Chart */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Performance Overview</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#E2E8F0" />
                  <PolarAngleAxis dataKey="metric" style={{ fontSize: '12px' }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar
                    name="Metrics"
                    dataKey="value"
                    stroke="#4F46E5"
                    fill="#4F46E5"
                    fillOpacity={0.3}
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Bar Chart */}
            {barData.length > 0 && (
              <div className="bg-white rounded-lg border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-900 mb-4">Historical Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={barData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="name" style={{ fontSize: '12px' }} />
                    <YAxis style={{ fontSize: '12px' }} />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill="#10B981" name="Accuracy %" />
                    <Bar dataKey="f1Score" fill="#4F46E5" name="F1 Score %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </>
      )}

      {/* Evaluation History */}
      <div className="bg-white rounded-lg border border-slate-200">
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">Evaluation History</h2>
        </div>
        <div className="p-6">
          {evaluations.length === 0 ? (
            <div className="text-center py-12">
              <BarChart3 className="h-16 w-16 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-600 mb-2">No evaluations yet</p>
              <p className="text-sm text-slate-500">
                Run your first evaluation to see results here
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200">
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Model</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Accuracy</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Precision</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Recall</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">F1 Score</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Perplexity</th>
                    <th className="text-left py-2 px-3 text-slate-600 font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluations.map((eval, idx) => (
                    <tr key={eval.id} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="py-2 px-3">{eval.model_id}</td>
                      <td className="py-2 px-3 font-mono text-green-600">
                        {(eval.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 font-mono">
                        {(eval.precision * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 font-mono">
                        {(eval.recall * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 font-mono text-indigo-600">
                        {(eval.f1_score * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 font-mono">{eval.perplexity.toFixed(2)}</td>
                      <td className="py-2 px-3 text-xs text-slate-600">
                        {new Date(eval.evaluated_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Evaluation;