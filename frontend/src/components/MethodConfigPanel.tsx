import React from 'react';
import { Input } from '../ui/input.jsx';
import { Label } from '../ui/label.jsx';
import { Slider } from '../ui/slider.jsx';
import { Switch } from '../ui/switch.jsx';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert.jsx';
import { Info, Zap, ChevronDown, ChevronUp, Database, Settings } from 'lucide-react';
import { TrainingMethod } from './TrainingMethodSelector';

export interface MethodSpecificConfig {
  // DoRA
  use_dora?: boolean;
  dora_simple?: boolean;

  // LoRA+
  lora_plus_ratio?: number;

  // IA³
  ia3_target_modules?: string[];
  ia3_feedforward_modules?: string[];

  // VeRA
  vera_rank?: number;
  vera_seed?: number;

  // AdaLoRA
  adalora_init_r?: number;
  adalora_target_r?: number;
  adalora_tinit?: number;
  adalora_tfinal?: number;
  adalora_deltaT?: number;
  adalora_beta1?: number;
  adalora_beta2?: number;

  // OFT
  oft_r?: number;
  oft_dropout?: number;
  oft_init_weights?: boolean;
}

interface MethodConfigPanelProps {
  method: TrainingMethod;
  config: MethodSpecificConfig;
  onConfigChange: (config: MethodSpecificConfig) => void;
}

export function MethodConfigPanel({ method, config, onConfigChange }: MethodConfigPanelProps) {
  const renderDoRAConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-blue-50 border-blue-200">
        <Info className="h-4 w-4 text-blue-600" />
        <AlertTitle className="text-blue-800">Weight-Decomposed Low-Rank Adaptation</AlertTitle>
        <AlertDescription className="text-blue-700">
          DoRA memisahkan update bobot menjadi magnitude dan direction, menghasilkan stabilitas training lebih baik dan performa SOTA.
        </AlertDescription>
      </Alert>

      <div className="flex items-center justify-between p-4 border rounded-lg bg-white">
        <div className="space-y-0.5">
          <Label htmlFor="use_dora" className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Enable DoRA
          </Label>
          <p className="text-sm text-muted-foreground">
            Aktifkan weight decomposition untuk performa lebih baik
          </p>
        </div>
        <Switch
          id="use_dora"
          checked={config.use_dora ?? true}
          onCheckedChange={(checked: boolean) => onConfigChange({ ...config, use_dora: checked })}
        />
      </div>

      <div className="flex items-center justify-between p-4 border rounded-lg bg-white">
        <div className="space-y-0.5">
          <Label htmlFor="dora_simple" className="flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Simplified Mode
          </Label>
          <p className="text-sm text-muted-foreground">
            Mode sederhana untuk training sedikit lebih cepat (minimal quality trade-off)
          </p>
        </div>
        <Switch
          id="dora_simple"
          checked={config.dora_simple ?? false}
          onCheckedChange={(checked: boolean) => onConfigChange({ ...config, dora_simple: checked })}
        />
      </div>
    </div>
  );

  const renderLoRAPlusConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-green-50 border-green-200">
        <Info className="h-4 w-4 text-green-600" />
        <AlertTitle className="text-green-800">Layer-wise Learning Rates</AlertTitle>
        <AlertDescription className="text-green-700">
          LoRA+ menggunakan learning rate yang lebih tinggi untuk matriks A. Rasio 16x memberikan 2x kecepatan convergence dengan kualitas sama.
        </AlertDescription>
      </Alert>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Learning Rate Ratio (A:B)
          </Label>
          <span className="text-lg font-semibold">{config.lora_plus_ratio ?? 16}x</span>
        </div>
        <Slider
          value={[config.lora_plus_ratio ?? 16]}
          onValueChange={([v]: number[]) => onConfigChange({ ...config, lora_plus_ratio: v })}
          min={1}
          max={64}
          step={1}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Conservative (1x)</span>
          <span>Standard (16x)</span>
          <span>Aggressive (64x)</span>
        </div>
        <p className="text-sm text-muted-foreground pt-2">
          Higher ratio = faster convergence, but potentially less stable. 16x is recommended for most cases.
        </p>
      </div>
    </div>
  );

  const renderIA3Config = () => (
    <div className="space-y-6">
      <Alert className="bg-purple-50 border-purple-200">
        <Info className="h-4 w-4 text-purple-600" />
        <AlertTitle className="text-purple-800">Infused Adapter (IA³)</AlertTitle>
        <AlertDescription className="text-purple-700">
          IA³ menggunakan element-wise scaling vectors. Lebih sedikit parameter dari LoRA dan inference lebih cepat.
        </AlertDescription>
      </Alert>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <Label className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          Target Modules (Comma-separated)
        </Label>
        <Input
          value={(config.ia3_target_modules ?? ['k_proj', 'v_proj', 'down_proj']).join(', ')}
          onChange={(e) =>
            onConfigChange({
              ...config,
              ia3_target_modules: e.target.value.split(',').map((s: string) => s.trim()),
            })
          }
          placeholder="k_proj, v_proj, down_proj"
        />
        <p className="text-xs text-muted-foreground">
          Recommended: k_proj, v_proj untuk attention; tambahkan down_proj untuk feedforward
        </p>
      </div>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <Label className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          Feedforward Modules (Comma-separated)
        </Label>
        <Input
          value={(config.ia3_feedforward_modules ?? ['down_proj']).join(', ')}
          onChange={(e) =>
            onConfigChange({
              ...config,
              ia3_feedforward_modules: e.target.value.split(',').map((s: string) => s.trim()),
            })
          }
          placeholder="down_proj"
        />
        <p className="text-xs text-muted-foreground">
          Must be subset of target modules. Usually down_proj for feedforward adaptation.
        </p>
      </div>
    </div>
  );

  const renderVeRAConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-indigo-50 border-indigo-200">
        <Info className="h-4 w-4 text-indigo-600" />
        <AlertTitle className="text-indigo-800">Vector-based Random Matrix Adaptation</AlertTitle>
        <AlertDescription className="text-indigo-700">
          VeRA menggunakan random frozen projections dan hanya meng-train scaling vectors. ~10,000x lebih sedikit parameter dari LoRA - ideal untuk edge deployment!
        </AlertDescription>
      </Alert>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Projection Rank (Frozen)
          </Label>
          <span className="text-lg font-semibold">{config.vera_rank ?? 256}</span>
        </div>
        <Slider
          value={[config.vera_rank ?? 256]}
          onValueChange={([v]: number[]) => onConfigChange({ ...config, vera_rank: v })}
          min={64}
          max={1024}
          step={64}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Minimal (64)</span>
          <span>Standard (256)</span>
          <span>Large (1024)</span>
        </div>
        <p className="text-sm text-muted-foreground pt-2">
          Higher rank = more capacity, but matrices are frozen so no extra trainable parameters.
        </p>
      </div>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <Label className="flex items-center gap-2">
          <Settings className="w-4 h-4" />
          Random Seed (Reproducibility)
        </Label>
        <Input
          type="number"
          value={config.vera_seed ?? 42}
          onChange={(e) => onConfigChange({ ...config, vera_seed: parseInt(e.target.value) })}
          min={0}
          max={999999}
        />
        <p className="text-xs text-muted-foreground">
          Seed untuk reproducible random initialization dari frozen matrices.
        </p>
      </div>
    </div>
  );

  const renderAdaLoRAConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-amber-50 border-amber-200">
        <Info className="h-4 w-4 text-amber-600" />
        <AlertTitle className="text-amber-800">Adaptive Budget Allocation</AlertTitle>
        <AlertDescription className="text-amber-700">
          AdaLoRA menggunakan SVD dan mengalokasikan budget parameter secara dinamis. Budget akan dipruning dari initial rank ke target rank selama training.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 border rounded-lg bg-white space-y-4">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Initial Rank
            </Label>
            <span className="text-lg font-semibold">{config.adalora_init_r ?? 12}</span>
          </div>
          <Slider
            value={[config.adalora_init_r ?? 12]}
            onValueChange={([v]: number[]) => onConfigChange({ ...config, adalora_init_r: v })}
            min={1}
            max={64}
            step={1}
          />
        </div>

        <div className="p-4 border rounded-lg bg-white space-y-4">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Target Budget Rank
            </Label>
            <span className="text-lg font-semibold">{config.adalora_target_r ?? 4}</span>
          </div>
          <Slider
            value={[config.adalora_target_r ?? 4]}
            onValueChange={([v]: number[]) => onConfigChange({ ...config, adalora_target_r: v })}
            min={1}
            max={32}
            step={1}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-lg bg-white space-y-2">
          <Label>Warmup Steps (tinit)</Label>
          <Input
            type="number"
            value={config.adalora_tinit ?? 0}
            onChange={(e) => onConfigChange({ ...config, adalora_tinit: parseInt(e.target.value) })}
          />
        </div>
        <div className="p-4 border rounded-lg bg-white space-y-2">
          <Label>Pruning End (tfinal)</Label>
          <Input
            type="number"
            value={config.adalora_tfinal ?? 1000}
            onChange={(e) => onConfigChange({ ...config, adalora_tfinal: parseInt(e.target.value) })}
          />
        </div>
        <div className="p-4 border rounded-lg bg-white space-y-2">
          <Label>Pruning Interval (deltaT)</Label>
          <Input
            type="number"
            value={config.adalora_deltaT ?? 10}
            onChange={(e) => onConfigChange({ ...config, adalora_deltaT: parseInt(e.target.value) })}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 border rounded-lg bg-white space-y-4">
          <Label>Beta1 (Importance Decay)</Label>
          <Slider
            value={[config.adalora_beta1 ?? 0.85]}
            onValueChange={([v]: number[]) => onConfigChange({ ...config, adalora_beta1: v })}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
        <div className="p-4 border rounded-lg bg-white space-y-4">
          <Label>Beta2 (Second Moment Decay)</Label>
          <Slider
            value={[config.adalora_beta2 ?? 0.85]}
            onValueChange={([v]: number[]) => onConfigChange({ ...config, adalora_beta2: v })}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
      </div>
    </div>
  );

  const renderOFTConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-rose-50 border-rose-200">
        <Info className="h-4 w-4 text-rose-600" />
        <AlertTitle className="text-rose-800">Orthogonal Fine-Tuning</AlertTitle>
        <AlertDescription className="text-rose-700">
          OFT menggunakan orthogonal transformations dengan multiplicative updates. Sangat stabil dan baik untuk multimodal tasks.
        </AlertDescription>
      </Alert>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            OFT Rank
          </Label>
          <span className="text-lg font-semibold">{config.oft_r ?? 8}</span>
        </div>
        <Slider
          value={[config.oft_r ?? 8]}
          onValueChange={([v]: number[]) => onConfigChange({ ...config, oft_r: v })}
          min={1}
          max={64}
          step={1}
        />
      </div>

      <div className="p-4 border rounded-lg bg-white space-y-4">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Module Dropout
          </Label>
          <span className="text-lg font-semibold">{((config.oft_dropout ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <Slider
          value={[config.oft_dropout ?? 0]}
          onValueChange={([v]: number[]) => onConfigChange({ ...config, oft_dropout: v })}
          min={0}
          max={0.5}
          step={0.01}
        />
      </div>

      <div className="flex items-center justify-between p-4 border rounded-lg bg-white">
        <div className="space-y-0.5">
          <Label htmlFor="oft_init_weights" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Initialize Weights
          </Label>
          <p className="text-sm text-muted-foreground">
            Initialize dengan orthogonal matrices
          </p>
        </div>
        <Switch
          id="oft_init_weights"
          checked={config.oft_init_weights ?? true}
          onCheckedChange={(checked: boolean) => onConfigChange({ ...config, oft_init_weights: checked })}
        />
      </div>
    </div>
  );

  const renderQLoRAConfig = () => (
    <div className="space-y-6">
      <Alert className="bg-gray-50 border-gray-200">
        <Info className="h-4 w-4 text-gray-600" />
        <AlertTitle className="text-gray-800">QLoRA (Classic)</AlertTitle>
        <AlertDescription className="text-gray-700">
          QLoRA adalah metode standard dengan 4-bit quantization dan Low-Rank Adaptation. Metode yang paling mature dan stable.
        </AlertDescription>
      </Alert>

      <div className="p-4 border rounded-lg bg-white">
        <p className="text-sm text-muted-foreground">
          Tidak ada konfigurasi method-specific untuk QLoRA. Semua pengaturan menggunakan konfigurasi LoRA standard.
        </p>
      </div>
    </div>
  );

  const renderers: Record<TrainingMethod, () => React.ReactNode> = {
    dora: renderDoRAConfig,
    qlora: renderQLoRAConfig,
    ia3: renderIA3Config,
    vera: renderVeRAConfig,
    lora_plus: renderLoRAPlusConfig,
    adalora: renderAdaLoRAConfig,
    oft: renderOFTConfig,
  };

  const renderer = renderers[method];

  if (!renderer) {
    return (
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          No additional configuration needed for {method.toUpperCase()}.
        </AlertDescription>
      </Alert>
    );
  }

  return <div className="space-y-4">{renderer()}</div>;
}
