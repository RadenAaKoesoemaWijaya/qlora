import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import { 
  Activity, 
  StopCircle, 
  CheckCircle, 
  AlertCircle,
  TrendingDown,
  Clock,
  Cpu,
  Play,
  Pause
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrainingMonitor = () => {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(jobId);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJobs();
  }, []);

  useEffect(() => {
    if (selectedJobId) {
      fetchJobData();
      const interval = setInterval(fetchJobData, 2000); // Update every 2 seconds
      return () => clearInterval(interval);
    }
  }, [selectedJobId]);

  const fetchJobs = async () => {
    try {
      const response = await axios.get(`${API}/training/jobs`);
      setJobs(response.data);
      if (!selectedJobId && response.data.length > 0) {
        setSelectedJobId(response.data[0].id);
      }
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    }
  };

  const fetchJobData = async () => {
    try {
      const [jobRes, metricsRes] = await Promise.all([
        axios.get(`${API}/training/jobs/${selectedJobId}`),
        axios.get(`${API}/training/jobs/${selectedJobId}/metrics`)
      ]);
      setJob(jobRes.data);
      setMetrics(metricsRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch job data:', error);
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    if (!window.confirm('Are you sure you want to stop this training?')) return;

    try {
      await axios.post(`${API}/training/jobs/${selectedJobId}/stop`);
      fetchJobData();
    } catch (error) {
      console.error('Failed to stop training:', error);
      alert('Failed to stop training');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'training':
      case 'initializing':
        return 'text-blue-600 bg-blue-100';
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'failed':
        return 'text-red-600 bg-red-100';
      case 'stopped':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'training':
      case 'initializing':
        return <Activity className="h-5 w-5 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5" />;
      case 'stopped':
        return <StopCircle className="h-5 w-5" />;
      default:
        return <Clock className="h-5 w-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse flex items-center space-x-2">
          <Activity className="h-6 w-6 text-indigo-600 animate-spin" />
          <span className="text-slate-600">Loading training monitor...</span>
        </div>
      </div>
    );
  }

  if (!job && jobs.length === 0) {
    return (
      <div className="text-center py-12" data-testid="no-training-jobs">
        <Activity className="h-16 w-16 text-slate-300 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-slate-900 mb-2">No Training Jobs</h2>
        <p className="text-slate-600 mb-6">Start a training job to monitor its progress here</p>
        <button
          onClick={() => navigate('/training/configure')}
          className="px-6 py-2.5 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 transition-colors"
        >
          Configure Training
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="training-monitor-page">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Training Monitor</h1>
          <p className="text-slate-600 mt-1">Real-time monitoring of fine-tuning progress</p>
        </div>
        {job && (job.status === 'training' || job.status === 'initializing') && (
          <button
            data-testid="stop-training-button"
            onClick={handleStopTraining}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors flex items-center space-x-2"
          >
            <StopCircle className="h-5 w-5" />
            <span>Stop Training</span>
          </button>
        )}
      </div>

      {/* Job Selector */}
      {jobs.length > 1 && (
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Select Training Job
          </label>
          <select
            data-testid="job-selector"
            value={selectedJobId}
            onChange={(e) => setSelectedJobId(e.target.value)}
            className="w-full md:w-96 px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {jobs.map(j => (
              <option key={j.id} value={j.id}>
                {j.model_name} - {j.status} - {new Date(j.started_at).toLocaleString()}
              </option>
            ))}
          </select>
        </div>
      )}

      {job && (
        <>
          {/* Status Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Status */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-600">Status</p>
                  <div className={`flex items-center space-x-2 mt-2 px-3 py-1.5 rounded-full ${getStatusColor(job.status)}`}>
                    {getStatusIcon(job.status)}
                    <span className="text-sm font-medium capitalize">{job.status}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Progress */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <p className="text-sm font-medium text-slate-600">Progress</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {job.progress.toFixed(1)}%
              </p>
              <div className="w-full bg-slate-200 rounded-full h-2 mt-3">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>

            {/* Epoch */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <p className="text-sm font-medium text-slate-600">Epoch</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {job.current_epoch}/{job.total_epochs}
              </p>
              <p className="text-sm text-slate-500 mt-1">Training iterations</p>
            </div>

            {/* Current Loss */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <p className="text-sm font-medium text-slate-600">Current Loss</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {job.current_loss ? job.current_loss.toFixed(4) : '-'}
              </p>
              <div className="flex items-center space-x-1 text-sm text-green-600 mt-1">
                <TrendingDown className="h-4 w-4" />
                <span>Decreasing</span>
              </div>
            </div>
          </div>

          {/* Training Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Model & Dataset Info */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Training Details</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-xs text-slate-500">Model</p>
                  <p className="text-sm font-medium text-slate-900">{job.model_name}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Dataset</p>
                  <p className="text-sm font-medium text-slate-900">{job.dataset_name}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Started At</p>
                  <p className="text-sm font-medium text-slate-900">
                    {new Date(job.started_at).toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Learning Rate</p>
                  <p className="text-sm font-medium text-slate-900">{job.learning_rate}</p>
                </div>
              </div>
            </div>

            {/* Configuration */}
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Configuration</h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-slate-500">LoRA Rank</p>
                  <p className="text-sm font-medium text-slate-900">{job.config.lora_rank}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">LoRA Alpha</p>
                  <p className="text-sm font-medium text-slate-900">{job.config.lora_alpha}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Batch Size</p>
                  <p className="text-sm font-medium text-slate-900">{job.config.batch_size}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Max Seq Length</p>
                  <p className="text-sm font-medium text-slate-900">{job.config.max_seq_length}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Compute</p>
                  <p className="text-sm font-medium text-slate-900 flex items-center">
                    <Cpu className="h-3 w-3 mr-1" />
                    {job.config.use_gpu ? 'GPU' : 'CPU'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Grad Accum</p>
                  <p className="text-sm font-medium text-slate-900">
                    {job.config.gradient_accumulation_steps}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Loss Chart */}
          {metrics.length > 0 && (
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={metrics}>
                  <defs>
                    <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#4F46E5" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis 
                    dataKey="step" 
                    stroke="#64748B"
                    style={{ fontSize: '12px' }}
                  />
                  <YAxis 
                    stroke="#64748B"
                    style={{ fontSize: '12px' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#FFF', 
                      border: '1px solid #E2E8F0',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#4F46E5" 
                    strokeWidth={2}
                    fill="url(#lossGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Metrics Table */}
          {metrics.length > 0 && (
            <div className="bg-white rounded-lg border border-slate-200 p-6">
              <h3 className="font-semibold text-slate-900 mb-4">Recent Metrics</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200">
                      <th className="text-left py-2 px-3 text-slate-600 font-medium">Step</th>
                      <th className="text-left py-2 px-3 text-slate-600 font-medium">Epoch</th>
                      <th className="text-left py-2 px-3 text-slate-600 font-medium">Loss</th>
                      <th className="text-left py-2 px-3 text-slate-600 font-medium">Learning Rate</th>
                      <th className="text-left py-2 px-3 text-slate-600 font-medium">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.slice(-10).reverse().map((metric, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                        <td className="py-2 px-3 font-mono">{metric.step}</td>
                        <td className="py-2 px-3">{metric.epoch}</td>
                        <td className="py-2 px-3 font-mono text-indigo-600">{metric.loss.toFixed(4)}</td>
                        <td className="py-2 px-3 font-mono text-xs">{metric.learning_rate}</td>
                        <td className="py-2 px-3 text-xs text-slate-600">
                          {new Date(metric.timestamp).toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default TrainingMonitor;
