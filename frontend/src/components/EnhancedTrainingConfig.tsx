import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card.jsx';
import { Button } from '../ui/button.jsx';
import { Input } from '../ui/input.jsx';
import { Label } from '../ui/label.jsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select.jsx';
import { Slider } from '../ui/slider.jsx';
import { Switch } from '../ui/switch.jsx';
import { Textarea } from '../ui/textarea.jsx';
import { Badge } from '../ui/badge.jsx';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert.jsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs.jsx';
import { 
  Collapsible, 
  CollapsibleContent, 
  CollapsibleTrigger 
} from '../ui/collapsible.jsx';
import { 
  ChevronDown, 
  ChevronUp, 
  Info, 
  AlertTriangle, 
  CheckCircle,
  Settings,
  Cpu,
  Database,
  Zap,
  Shield,
  Clock,
  Save,
  RotateCcw
} from 'lucide-react';

// Types for training configuration
type ModelSize = 'small' | 'medium' | 'large' | 'xl';
type QuantizationType = '4bit' | '8bit' | '16bit' | '32bit';
type OptimizerType = 'adamw' | 'adam' | 'sgd' | 'adamw_8bit';
type SchedulerType = 'linear' | 'cosine' | 'polynomial' | 'constant';
type DatasetFormat = 'json' | 'jsonl' | 'csv' | 'txt' | 'parquet' | 'xlsx';

interface TrainingConfig {
  // Basic Settings
  job_name: string;
  model_name: string;
  model_size: ModelSize;
  dataset_name: string;
  dataset_format: DatasetFormat;
  
  // Training Parameters
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  max_seq_length: number;
  
  // LoRA Configuration
  lora_rank: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  
  // Quantization
  quantization: QuantizationType;
  use_gradient_checkpointing: boolean;
  
  // Optimizer & Scheduler
  optimizer: OptimizerType;
  scheduler: SchedulerType;
  weight_decay: number;
  
  // Advanced Settings
  gradient_accumulation_steps: number;
  max_grad_norm: number;
  save_steps: number;
  logging_steps: number;
  evaluation_strategy: 'steps' | 'epoch' | 'no';
  eval_steps: number;
  load_best_model_at_end: boolean;
  metric_for_best_model: string;
  
  // System Settings
  gpu_device: string;
  mixed_precision: boolean;
  dataloader_num_workers: number;
  remove_unused_columns: boolean;
  
  // Custom Settings
  custom_prompt_template: string;
  custom_completion_template: string;
  system_message: string;
}

interface TrainingPreset {
  name: string;
  description: string;
  config: Partial<TrainingConfig>;
  icon: React.ReactNode;
  color: string;
}

interface EnhancedTrainingConfigProps {
  onConfigChange: (config: TrainingConfig) => void;
  onSavePreset: (preset: Omit<TrainingPreset, 'icon' | 'color'>) => void;
  initialConfig?: Partial<TrainingConfig>;
  availableModels: string[];
  availableDatasets: string[];
  availableGpuDevices: string[];
  isTrainingActive?: boolean;
}

const defaultConfig: TrainingConfig = {
  job_name: '',
  model_name: 'microsoft/DialoGPT-medium',
  model_size: 'medium',
  dataset_name: '',
  dataset_format: 'json',
  num_epochs: 3,
  batch_size: 4,
  learning_rate: 5e-5,
  warmup_steps: 100,
  max_seq_length: 512,
  lora_rank: 16,
  lora_alpha: 32,
  lora_dropout: 0.1,
  target_modules: ['q_proj', 'v_proj'],
  quantization: '16bit',
  use_gradient_checkpointing: true,
  optimizer: 'adamw_8bit',
  scheduler: 'linear',
  weight_decay: 0.01,
  gradient_accumulation_steps: 1,
  max_grad_norm: 1.0,
  save_steps: 500,
  logging_steps: 10,
  evaluation_strategy: 'steps',
  eval_steps: 500,
  load_best_model_at_end: true,
  metric_for_best_model: 'eval_loss',
  gpu_device: 'auto',
  mixed_precision: true,
  dataloader_num_workers: 0,
  remove_unused_columns: true,
  custom_prompt_template: '### Instruction: {prompt}\n\n### Response:',
  custom_completion_template: '{completion}',
  system_message: 'You are a helpful AI assistant.'
};

const trainingPresets: TrainingPreset[] = [
  {
    name: 'Quick Start',
    description: 'Konfigurasi cepat untuk percobaan awal',
    config: {
      num_epochs: 1,
      batch_size: 2,
      learning_rate: 1e-4,
      lora_rank: 8,
      max_seq_length: 256
    },
    icon: <Zap className="h-4 w-4" />,
    color: 'text-green-600 bg-green-100'
  },
  {
    name: 'Balanced',
    description: 'Keseimbangan antara performa dan kualitas',
    config: {
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 5e-5,
      lora_rank: 16,
      max_seq_length: 512
    },
    icon: <Settings className="h-4 w-4" />,
    color: 'text-blue-600 bg-blue-100'
  },
  {
    name: 'High Quality',
    description: 'Kualitas maksimal untuk produksi',
    config: {
      num_epochs: 5,
      batch_size: 8,
      learning_rate: 2e-5,
      lora_rank: 32,
      max_seq_length: 1024,
      gradient_accumulation_steps: 2
    },
    icon: <Shield className="h-4 w-4" />,
    color: 'text-purple-600 bg-purple-100'
  },
  {
    name: 'Memory Efficient',
    description: 'Efisiensi memory untuk GPU terbatas',
    config: {
      quantization: '8bit',
      use_gradient_checkpointing: true,
      optimizer: 'adamw_8bit',
      batch_size: 1,
      gradient_accumulation_steps: 4,
      max_seq_length: 256
    },
    icon: <Cpu className="h-4 w-4" />,
    color: 'text-orange-600 bg-orange-100'
  }
];

export default function EnhancedTrainingConfig({
  onConfigChange,
  onSavePreset,
  initialConfig,
  availableModels,
  availableDatasets,
  availableGpuDevices,
  isTrainingActive = false
}: EnhancedTrainingConfigProps) {
  const [config, setConfig] = useState<TrainingConfig>({
    ...defaultConfig,
    ...initialConfig
  });
  const [activeTab, setActiveTab] = useState('basic');
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    basic: true,
    training: true,
    lora: false,
    advanced: false,
    system: false
  });
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [presetName, setPresetName] = useState('');
  const [presetDescription, setPresetDescription] = useState('');

  useEffect(() => {
    validateConfig();
    onConfigChange(config);
  }, [config]);

  const validateConfig = () => {
    const errors: Record<string, string> = {};

    if (!config.job_name.trim()) {
      errors.job_name = 'Nama job harus diisi';
    }

    if (!config.dataset_name) {
      errors.dataset_name = 'Dataset harus dipilih';
    }

    if (config.learning_rate <= 0 || config.learning_rate > 1) {
      errors.learning_rate = 'Learning rate harus antara 0 dan 1';
    }

    if (config.batch_size < 1) {
      errors.batch_size = 'Batch size minimal 1';
    }

    if (config.num_epochs < 1) {
      errors.num_epochs = 'Jumlah epochs minimal 1';
    }

    if (config.lora_rank < 1 || config.lora_rank > 64) {
      errors.lora_rank = 'LoRA rank harus antara 1 dan 64';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const updateConfig = (updates: Partial<TrainingConfig>) => {
    setConfig(prev => ({ ...prev, ...updates }));
  };

  const applyPreset = (preset: TrainingPreset) => {
    updateConfig(preset.config);
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleSavePreset = () => {
    if (!presetName.trim()) return;

    onSavePreset({
      name: presetName,
      description: presetDescription,
      config
    });

    setShowSaveDialog(false);
    setPresetName('');
    setPresetDescription('');
  };

  const resetToDefault = () => {
    setConfig(defaultConfig);
  };

  const SectionHeader = ({ title, section, icon }: { title: string; section: string; icon: React.ReactNode }) => (
    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors">
      <div className="flex items-center space-x-3" onClick={() => toggleSection(section)}>
        {icon}
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
      </div>
      {expandedSections[section] ? (
        <ChevronUp className="h-5 w-5 text-gray-500" />
      ) : (
        <ChevronDown className="h-5 w-5 text-gray-500" />
      )}
    </div>
  );

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-2">
            <Settings className="h-8 w-8 text-blue-600" />
            <span>Konfigurasi Training</span>
          </h1>
          <p className="text-gray-600 mt-2">Atur parameter training model QLoRA dengan mudah</p>
        </div>
        <div className="flex space-x-3">
          <Button
            variant="outline"
            onClick={resetToDefault}
            disabled={isTrainingActive}
            className="flex items-center space-x-2"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </Button>
          <Button
            onClick={() => setShowSaveDialog(true)}
            disabled={isTrainingActive}
            className="flex items-center space-x-2"
          >
            <Save className="h-4 w-4" />
            <span>Simpan Preset</span>
          </Button>
        </div>
      </div>

      {/* Training Presets */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-yellow-600" />
            <span>Preset Training</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {trainingPresets.map((preset) => (
              <div
                key={preset.name}
                className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                  'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => applyPreset(preset)}
              >
                <div className={`flex items-center space-x-2 mb-2 ${preset.color}`}>
                  {preset.icon}
                  <h3 className="font-semibold">{preset.name}</h3>
                </div>
                <p className="text-sm text-gray-600">{preset.description}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Validation Alerts */}
      {Object.keys(validationErrors).length > 0 && (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Validasi Error</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside mt-2">
              {Object.values(validationErrors).map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Configuration Sections */}
      <div className="space-y-6">
        {/* Basic Settings */}
        <Card>
          <CardHeader>
            <SectionHeader title="Pengaturan Dasar" section="basic" icon={<Database className="h-5 w-5 text-blue-600" />} />
          </CardHeader>
          {expandedSections.basic && (
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="job_name">Nama Job *</Label>
                  <Input
                    id="job_name"
                    value={config.job_name}
                    onChange={(e) => updateConfig({ job_name: e.target.value })}
                    placeholder="contoh: fine-tuning-model-saya"
                    className={validationErrors.job_name ? 'border-red-500' : ''}
                    disabled={isTrainingActive}
                  />
                  {validationErrors.job_name && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.job_name}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="model_name">Model *</Label>
                  <Select
                    value={config.model_name}
                    onValueChange={(value) => updateConfig({ model_name: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="model_name" className={validationErrors.model_name ? 'border-red-500' : ''}>
                      <SelectValue placeholder="Pilih model" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableModels.map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="dataset_name">Dataset *</Label>
                  <Select
                    value={config.dataset_name}
                    onValueChange={(value) => updateConfig({ dataset_name: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="dataset_name" className={validationErrors.dataset_name ? 'border-red-500' : ''}>
                      <SelectValue placeholder="Pilih dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableDatasets.map((dataset) => (
                        <SelectItem key={dataset} value={dataset}>
                          {dataset}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {validationErrors.dataset_name && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.dataset_name}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="dataset_format">Format Dataset</Label>
                  <Select
                    value={config.dataset_format}
                    onValueChange={(value: DatasetFormat) => updateConfig({ dataset_format: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="dataset_format">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="json">JSON</SelectItem>
                      <SelectItem value="jsonl">JSONL</SelectItem>
                      <SelectItem value="csv">CSV</SelectItem>
                      <SelectItem value="txt">TXT</SelectItem>
                      <SelectItem value="parquet">Parquet</SelectItem>
                      <SelectItem value="xlsx">XLSX</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          )}
        </Card>

        {/* Training Parameters */}
        <Card>
          <CardHeader>
            <SectionHeader title="Parameter Training" section="training" icon={<Cpu className="h-5 w-5 text-green-600" />} />
          </CardHeader>
          {expandedSections.training && (
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="num_epochs">Jumlah Epochs</Label>
                  <Input
                    id="num_epochs"
                    type="number"
                    min="1"
                    max="100"
                    value={config.num_epochs}
                    onChange={(e) => updateConfig({ num_epochs: parseInt(e.target.value) || 1 })}
                    className={validationErrors.num_epochs ? 'border-red-500' : ''}
                    disabled={isTrainingActive}
                  />
                  {validationErrors.num_epochs && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.num_epochs}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="batch_size">Batch Size</Label>
                  <Input
                    id="batch_size"
                    type="number"
                    min="1"
                    max="64"
                    value={config.batch_size}
                    onChange={(e) => updateConfig({ batch_size: parseInt(e.target.value) || 1 })}
                    className={validationErrors.batch_size ? 'border-red-500' : ''}
                    disabled={isTrainingActive}
                  />
                  {validationErrors.batch_size && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.batch_size}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="learning_rate">Learning Rate</Label>
                  <Input
                    id="learning_rate"
                    type="number"
                    step="0.00001"
                    min="0.00001"
                    max="1"
                    value={config.learning_rate}
                    onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) || 0.00001 })}
                    className={validationErrors.learning_rate ? 'border-red-500' : ''}
                    disabled={isTrainingActive}
                  />
                  {validationErrors.learning_rate && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.learning_rate}</p>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="optimizer">Optimizer</Label>
                  <Select
                    value={config.optimizer}
                    onValueChange={(value: OptimizerType) => updateConfig({ optimizer: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="optimizer">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="adamw">AdamW</SelectItem>
                      <SelectItem value="adam">Adam</SelectItem>
                      <SelectItem value="sgd">SGD</SelectItem>
                      <SelectItem value="adamw_8bit">AdamW 8-bit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="scheduler">Scheduler</Label>
                  <Select
                    value={config.scheduler}
                    onValueChange={(value: SchedulerType) => updateConfig({ scheduler: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="scheduler">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="linear">Linear</SelectItem>
                      <SelectItem value="cosine">Cosine</SelectItem>
                      <SelectItem value="polynomial">Polynomial</SelectItem>
                      <SelectItem value="constant">Constant</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          )}
        </Card>

        {/* LoRA Configuration */}
        <Card>
          <CardHeader>
            <SectionHeader title="Konfigurasi LoRA" section="lora" icon={<Zap className="h-5 w-5 text-purple-600" />} />
          </CardHeader>
          {expandedSections.lora && (
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="lora_rank">LoRA Rank</Label>
                  <Input
                    id="lora_rank"
                    type="number"
                    min="1"
                    max="64"
                    value={config.lora_rank}
                    onChange={(e) => updateConfig({ lora_rank: parseInt(e.target.value) || 1 })}
                    className={validationErrors.lora_rank ? 'border-red-500' : ''}
                    disabled={isTrainingActive}
                  />
                  {validationErrors.lora_rank && (
                    <p className="text-red-500 text-sm mt-1">{validationErrors.lora_rank}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="lora_alpha">LoRA Alpha</Label>
                  <Input
                    id="lora_alpha"
                    type="number"
                    min="1"
                    max="128"
                    value={config.lora_alpha}
                    onChange={(e) => updateConfig({ lora_alpha: parseInt(e.target.value) || 1 })}
                    disabled={isTrainingActive}
                  />
                </div>

                <div>
                  <Label htmlFor="lora_dropout">LoRA Dropout</Label>
                  <Input
                    id="lora_dropout"
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={config.lora_dropout}
                    onChange={(e) => updateConfig({ lora_dropout: parseFloat(e.target.value) || 0 })}
                    disabled={isTrainingActive}
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="target_modules">Target Modules</Label>
                <Textarea
                  id="target_modules"
                  value={config.target_modules.join(', ')}
                  onChange={(e) => updateConfig({ target_modules: e.target.value.split(',').map(s => s.trim()).filter(s => s) })}
                  placeholder="q_proj, v_proj, k_proj, o_proj"
                  disabled={isTrainingActive}
                />
              </div>
            </CardContent>
          )}
        </Card>

        {/* Advanced Settings */}
        <Card>
          <CardHeader>
            <SectionHeader title="Pengaturan Lanjutan" section="advanced" icon={<Settings className="h-5 w-5 text-red-600" />} />
          </CardHeader>
          {expandedSections.advanced && (
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="quantization">Kuantilisasi</Label>
                  <Select
                    value={config.quantization}
                    onValueChange={(value: QuantizationType) => updateConfig({ quantization: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="quantization">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="4bit">4-bit</SelectItem>
                      <SelectItem value="8bit">8-bit</SelectItem>
                      <SelectItem value="16bit">16-bit</SelectItem>
                      <SelectItem value="32bit">32-bit</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label htmlFor="gpu_device">GPU Device</Label>
                  <Select
                    value={config.gpu_device}
                    onValueChange={(value) => updateConfig({ gpu_device: value })}
                    disabled={isTrainingActive}
                  >
                    <SelectTrigger id="gpu_device">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Otomatis</SelectItem>
                      {availableGpuDevices.map((device) => (
                        <SelectItem key={device} value={device}>
                          {device}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="use_gradient_checkpointing"
                    checked={config.use_gradient_checkpointing}
                    onCheckedChange={(checked) => updateConfig({ use_gradient_checkpointing: checked })}
                    disabled={isTrainingActive}
                  />
                  <Label htmlFor="use_gradient_checkpointing">Gradient Checkpointing</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    id="mixed_precision"
                    checked={config.mixed_precision}
                    onCheckedChange={(checked) => updateConfig({ mixed_precision: checked })}
                    disabled={isTrainingActive}
                  />
                  <Label htmlFor="mixed_precision">Mixed Precision</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Switch
                    id="remove_unused_columns"
                    checked={config.remove_unused_columns}
                    onCheckedChange={(checked) => updateConfig({ remove_unused_columns: checked })}
                    disabled={isTrainingActive}
                  />
                  <Label htmlFor="remove_unused_columns">Remove Unused Columns</Label>
                </div>
              </div>
            </CardContent>
          )}
        </Card>

        {/* Custom Templates */}
        <Card>
          <CardHeader>
            <SectionHeader title="Template Kustom" section="system" icon={<Clock className="h-5 w-5 text-indigo-600" />} />
          </CardHeader>
          {expandedSections.system && (
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="custom_prompt_template">Template Prompt</Label>
                <Textarea
                  id="custom_prompt_template"
                  value={config.custom_prompt_template}
                  onChange={(e) => updateConfig({ custom_prompt_template: e.target.value })}
                  placeholder="### Instruction: {prompt}\n\n### Response:"
                  rows={3}
                  disabled={isTrainingActive}
                />
              </div>

              <div>
                <Label htmlFor="custom_completion_template">Template Completion</Label>
                <Textarea
                  id="custom_completion_template"
                  value={config.custom_completion_template}
                  onChange={(e) => updateConfig({ custom_completion_template: e.target.value })}
                  placeholder="{completion}"
                  rows={2}
                  disabled={isTrainingActive}
                />
              </div>

              <div>
                <Label htmlFor="system_message">Pesan Sistem</Label>
                <Textarea
                  id="system_message"
                  value={config.system_message}
                  onChange={(e) => updateConfig({ system_message: e.target.value })}
                  placeholder="You are a helpful AI assistant."
                  rows={2}
                  disabled={isTrainingActive}
                />
              </div>
            </CardContent>
          )}
        </Card>
      </div>

      {/* Save Preset Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>Simpan Preset Training</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="preset_name">Nama Preset</Label>
                <Input
                  id="preset_name"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  placeholder="contoh: Konfigurasi Produksi"
                />
              </div>
              <div>
                <Label htmlFor="preset_description">Deskripsi</Label>
                <Textarea
                  id="preset_description"
                  value={presetDescription}
                  onChange={(e) => setPresetDescription(e.target.value)}
                  placeholder="Jelaskan kegunaan preset ini"
                  rows={3}
                />
              </div>
              <div className="flex space-x-3">
                <Button onClick={handleSavePreset} className="flex-1">
                  Simpan
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowSaveDialog(false)}
                  className="flex-1"
                >
                  Batal
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}