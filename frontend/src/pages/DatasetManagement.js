import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  File, 
  Trash2, 
  CheckCircle, 
  AlertCircle,
  FileJson,
  FileText,
  Database as DatabaseIcon
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DatasetManagement = () => {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${API}/datasets`);
      setDatasets(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
      setLoading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', file.name);

    try {
      await axios.post(`${API}/datasets/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      fetchDatasets();
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Failed to upload dataset');
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/json': ['.json', '.jsonl'],
      'text/csv': ['.csv']
    },
    multiple: false
  });

  const handleDelete = async (datasetId) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;

    try {
      await axios.delete(`${API}/datasets/${datasetId}`);
      fetchDatasets();
    } catch (error) {
      console.error('Delete failed:', error);
      alert('Failed to delete dataset');
    }
  };

  const handleSelectDataset = (dataset) => {
    localStorage.setItem('selectedDataset', JSON.stringify(dataset));
    navigate('/models');
  };

  const getFileIcon = (fileType) => {
    switch (fileType) {
      case 'JSON':
      case 'JSONL':
        return <FileJson className="h-6 w-6 text-blue-600" />;
      case 'CSV':
        return <FileText className="h-6 w-6 text-green-600" />;
      default:
        return <File className="h-6 w-6 text-slate-600" />;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse">Loading datasets...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="dataset-management-page">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dataset Management</h1>
        <p className="text-slate-600 mt-1">
          Upload and manage medical training datasets for fine-tuning
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        data-testid="dataset-upload-dropzone"
        className={`${
          isDragActive
            ? 'border-indigo-600 bg-indigo-50'
            : 'border-slate-300 hover:border-indigo-400'
        } border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center">
          {uploading ? (
            <>
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4" />
              <p className="text-slate-700 font-medium">Uploading...</p>
            </>
          ) : (
            <>
              <Upload className="h-12 w-12 text-slate-400 mb-4" />
              <p className="text-slate-700 font-medium mb-2">
                {isDragActive ? 'Drop file here...' : 'Drag & drop dataset file here'}
              </p>
              <p className="text-sm text-slate-500">
                or click to browse • Supports JSON, JSONL, CSV
              </p>
            </>
          )}
        </div>
      </div>

      {/* Info Banner */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-yellow-900">
            <p className="font-medium mb-1">Dataset Format</p>
            <p className="text-yellow-800">
              Expected format: Each record should have 'instruction', 'input', and 'output' fields for medical scenarios.
              Example: {{"instruction": "Diagnose condition", "input": "Patient symptoms...", "output": "Diagnosis..."}}
            </p>
          </div>
        </div>
      </div>

      {/* Datasets List */}
      <div>
        <h2 className="text-xl font-semibold text-slate-900 mb-4">Uploaded Datasets</h2>
        {datasets.length === 0 ? (
          <div className="bg-white border border-slate-200 rounded-lg p-12 text-center">
            <DatabaseIcon className="h-16 w-16 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-600 mb-2">No datasets uploaded yet</p>
            <p className="text-sm text-slate-500">Upload your first medical training dataset to get started</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                data-testid={`dataset-card-${dataset.id}`}
                className="bg-white border border-slate-200 rounded-lg p-6 hover:border-indigo-300 hover:shadow-md transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4 flex-1">
                    <div className="w-12 h-12 bg-slate-100 rounded-lg flex items-center justify-center">
                      {getFileIcon(dataset.file_type)}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-slate-900">{dataset.name}</h3>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-slate-600">
                        <span>{dataset.file_type}</span>
                        <span>•</span>
                        <span>{formatFileSize(dataset.size)}</span>
                        <span>•</span>
                        <span>{dataset.rows} rows</span>
                        <span>•</span>
                        <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center space-x-2 mt-2">
                        <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full flex items-center">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Valid
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      data-testid={`select-dataset-${dataset.id}`}
                      onClick={() => handleSelectDataset(dataset)}
                      className="px-4 py-2 bg-indigo-900 text-white rounded-md hover:bg-indigo-800 transition-colors text-sm"
                    >
                      Select
                    </button>
                    <button
                      data-testid={`delete-dataset-${dataset.id}`}
                      onClick={() => handleDelete(dataset.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-md transition-colors"
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
  );
};

export default DatasetManagement;