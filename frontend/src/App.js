import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import ModelSelection from './pages/ModelSelection';
import DatasetManagement from './pages/DatasetManagement';
import TrainingConfiguration from './pages/TrainingConfiguration';
import TrainingMonitor from './pages/TrainingMonitor';
import CheckpointManager from './pages/CheckpointManager';
import Evaluation from './pages/Evaluation';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="models" element={<ModelSelection />} />
          <Route path="datasets" element={<DatasetManagement />} />
          <Route path="training/configure" element={<TrainingConfiguration />} />
          <Route path="training/monitor" element={<TrainingMonitor />} />
          <Route path="training/monitor/:jobId" element={<TrainingMonitor />} />
          <Route path="checkpoints" element={<CheckpointManager />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;