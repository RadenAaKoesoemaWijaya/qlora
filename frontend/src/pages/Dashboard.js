import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  Activity, 
  Database, 
  Save, 
  CheckCircle, 
  Clock,
  TrendingUp,
  Brain,
  AlertCircle
} from 'lucide-react';
import { Link } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsRes, jobsRes] = await Promise.all([
        axios.get(`${API}/dashboard/stats`),
        axios.get(`${API}/training/jobs`)
      ]);
      setStats(statsRes.data);
      setRecentJobs(jobsRes.data.slice(0, 5));
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'training':
      case 'initializing':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'stopped':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse flex items-center space-x-2">
          <Activity className="h-6 w-6 text-indigo-600 animate-spin" />
          <span className="text-slate-600">Loading dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="dashboard">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-600 mt-1">Overview of your medical LLM fine-tuning projects</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Training Jobs */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 card-hover">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Total Training Jobs</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {stats?.total_training_jobs || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
              <Brain className="h-6 w-6 text-indigo-600" />
            </div>
          </div>
        </div>

        {/* Active Jobs */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 card-hover">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Active Training</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {stats?.active_training_jobs || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <Activity className="h-6 w-6 text-blue-600 animate-pulse" />
            </div>
          </div>
        </div>

        {/* Completed Jobs */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 card-hover">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Completed</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {stats?.completed_training_jobs || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>

        {/* Datasets */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 card-hover">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-slate-600">Datasets</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                {stats?.total_datasets || 0}
              </p>
            </div>
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
              <Database className="h-6 w-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Training Jobs */}
      <div className="bg-white rounded-lg border border-slate-200">
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">Recent Training Jobs</h2>
        </div>
        <div className="p-6">
          {recentJobs.length === 0 ? (
            <div className="text-center py-12">
              <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-3" />
              <p className="text-slate-600 mb-4">No training jobs yet</p>
              <Link
                to="/training/configure"
                data-testid="start-training-button"
                className="inline-flex items-center px-4 py-2 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 transition-colors"
              >
                Start Your First Training
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {recentJobs.map((job) => (
                <Link
                  key={job.id}
                  to={`/training/monitor/${job.id}`}
                  data-testid={`training-job-${job.id}`}
                  className="block border border-slate-200 rounded-lg p-4 hover:border-indigo-300 hover:bg-indigo-50/50 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="font-medium text-slate-900">{job.model_name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 mt-1">Dataset: {job.dataset_name}</p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-slate-500">
                        <span>Epoch {job.current_epoch}/{job.total_epochs}</span>
                        {job.current_loss && (
                          <span>Loss: {job.current_loss.toFixed(4)}</span>
                        )}
                        <span>Started: {new Date(job.started_at).toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="ml-4">
                      {job.status === 'training' && (
                        <div className="text-right">
                          <div className="text-2xl font-bold text-indigo-900">
                            {job.progress.toFixed(0)}%
                          </div>
                          <div className="text-xs text-slate-500">Progress</div>
                        </div>
                      )}
                    </div>
                  </div>
                  {job.status === 'training' && (
                    <div className="mt-3">
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div
                          className="bg-indigo-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          to="/datasets"
          data-testid="quick-action-datasets"
          className="bg-white border border-slate-200 rounded-lg p-6 hover:border-indigo-300 hover:shadow-md transition-all card-hover"
        >
          <Database className="h-8 w-8 text-indigo-600 mb-3" />
          <h3 className="font-semibold text-slate-900 mb-1">Upload Dataset</h3>
          <p className="text-sm text-slate-600">Add medical training data</p>
        </Link>

        <Link
          to="/models"
          data-testid="quick-action-models"
          className="bg-white border border-slate-200 rounded-lg p-6 hover:border-indigo-300 hover:shadow-md transition-all card-hover"
        >
          <Brain className="h-8 w-8 text-indigo-600 mb-3" />
          <h3 className="font-semibold text-slate-900 mb-1">Select Model</h3>
          <p className="text-sm text-slate-600">Choose base LLM model</p>
        </Link>

        <Link
          to="/training/configure"
          data-testid="quick-action-training"
          className="bg-white border border-slate-200 rounded-lg p-6 hover:border-indigo-300 hover:shadow-md transition-all card-hover"
        >
          <TrendingUp className="h-8 w-8 text-indigo-600 mb-3" />
          <h3 className="font-semibold text-slate-900 mb-1">Start Training</h3>
          <p className="text-sm text-slate-600">Configure & begin fine-tuning</p>
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;