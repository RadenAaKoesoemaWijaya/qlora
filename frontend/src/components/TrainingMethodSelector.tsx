import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx';
import { Badge } from '@/components/ui/badge.jsx';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group.jsx';
import { Label } from '@/components/ui/label.jsx';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.jsx';
import { Zap, TrendingUp, Cpu, Award, Info, Layers, Target, GitBranch } from 'lucide-react';

export type TrainingMethod = 'qlora' | 'dora' | 'ia3' | 'vera' | 'lora_plus' | 'adalora' | 'oft';

export interface TrainingMethodInfo {
  id: TrainingMethod;
  name: string;
  description: string;
  efficiency: 'extreme' | 'very_high' | 'high' | 'medium';
  performance: 'state_of_the_art' | 'excellent' | 'good' | 'baseline';
  difficulty: 'easy' | 'medium' | 'complex';
  badges: string[];
  parameterReduction: string;
  memoryReduction: string;
  paper: string;
}

const trainingMethods: TrainingMethodInfo[] = [
  {
    id: 'dora',
    name: 'DoRA',
    description: 'Weight-Decomposed Low-Rank Adaptation - SOTA performance dengan stabilitas lebih baik',
    efficiency: 'high',
    performance: 'state_of_the_art',
    difficulty: 'easy',
    badges: ['SOTA', 'Recommended', 'Easy Setup'],
    parameterReduction: '99.9%',
    memoryReduction: '75%',
    paper: 'DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024)',
  },
  {
    id: 'qlora',
    name: 'QLoRA (Classic)',
    description: 'Quantized Low-Rank Adaptation - Metode standard dengan 4-bit quantization',
    efficiency: 'high',
    performance: 'excellent',
    difficulty: 'easy',
    badges: ['Proven', 'Stable', 'Default'],
    parameterReduction: '99.9%',
    memoryReduction: '75%',
    paper: 'QLoRA: Efficient Finetuning of Quantized LLMs (2023)',
  },
  {
    id: 'ia3',
    name: 'IA³',
    description: 'Infused Adapter - Element-wise scaling dengan parameter lebih sedikit dari LoRA',
    efficiency: 'very_high',
    performance: 'excellent',
    difficulty: 'easy',
    badges: ['Fast Inference', 'Low Memory'],
    parameterReduction: '99.95%',
    memoryReduction: '80%',
    paper: 'Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper',
  },
  {
    id: 'vera',
    name: 'VeRA',
    description: 'Vector-based Random Matrix Adaptation - Ultra-efficient dengan 1/1000 parameter LoRA',
    efficiency: 'extreme',
    performance: 'good',
    difficulty: 'easy',
    badges: ['Ultra Low-Param', 'Edge Ready', 'Extreme Efficiency'],
    parameterReduction: '99.999%',
    memoryReduction: '95%',
    paper: 'VeRA: Vector-based Random Matrix Adaptation (2024)',
  },
  {
    id: 'lora_plus',
    name: 'LoRA+',
    description: 'LoRA dengan Layer-wise Learning Rates - 2x convergence speed',
    efficiency: 'high',
    performance: 'excellent',
    difficulty: 'easy',
    badges: ['Fast Convergence', 'Same Quality', 'Easy Upgrade'],
    parameterReduction: '99.9%',
    memoryReduction: '75%',
    paper: 'LoRA+: Efficient Low Rank Adaptation of Large Models (2024)',
  },
  {
    id: 'adalora',
    name: 'AdaLoRA',
    description: 'Adaptive Budget Allocation - Parameter dialokasikan secara dinamis selama training',
    efficiency: 'high',
    performance: 'state_of_the_art',
    difficulty: 'medium',
    badges: ['Adaptive', 'Advanced', 'Budget-aware'],
    parameterReduction: '99.9%',
    memoryReduction: '75%',
    paper: 'AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning (ICLR 2023)',
  },
  {
    id: 'oft',
    name: 'OFT',
    description: 'Orthogonal Fine-Tuning - Multiplicative updates dengan orthogonal constraint',
    efficiency: 'high',
    performance: 'excellent',
    difficulty: 'medium',
    badges: ['Orthogonal', 'Geometric', 'Multimodal'],
    parameterReduction: '99.9%',
    memoryReduction: '75%',
    paper: 'Controlling Text-to-Image Diffusion by Orthogonal Finetuning (CVPR 2024)',
  },
];

interface TrainingMethodSelectorProps {
  selectedMethod: TrainingMethod;
  onMethodChange: (method: TrainingMethod) => void;
  disabled?: boolean;
}

const getEfficiencyIcon = (efficiency: string) => {
  switch (efficiency) {
    case 'extreme':
      return <Info className="w-4 h-4 text-purple-500" />;
    case 'very_high':
      return <Zap className="w-4 h-4 text-green-500" />;
    case 'high':
      return <Cpu className="w-4 h-4 text-blue-500" />;
    default:
      return <Cpu className="w-4 h-4 text-gray-500" />;
  }
};

const getPerformanceBadge = (performance: string) => {
  switch (performance) {
    case 'state_of_the_art':
      return (
        <Badge className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white">
          <Award className="w-3 h-3 mr-1" /> SOTA
        </Badge>
      );
    case 'excellent':
      return (
        <Badge className="bg-green-500 text-white">
          <TrendingUp className="w-3 h-3 mr-1" /> Excellent
        </Badge>
      );
    default:
      return <Badge variant="secondary">Good</Badge>;
  }
};

export function TrainingMethodSelector({
  selectedMethod,
  onMethodChange,
  disabled = false,
}: TrainingMethodSelectorProps) {
  return (
    <TooltipProvider>
      <RadioGroup
        value={selectedMethod}
        onValueChange={(value: string): void => onMethodChange(value as TrainingMethod)}
        disabled={disabled}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {trainingMethods.map((method: TrainingMethodInfo) => (
            <div key={method.id}>
              <RadioGroupItem
                value={method.id}
                id={method.id}
                className="peer sr-only"
                disabled={disabled}
              />
              <Label htmlFor={method.id} className="cursor-pointer block h-full">
                <Card
                  className={`transition-all hover:border-blue-400 h-full flex flex-col ${
                    selectedMethod === method.id
                      ? 'border-blue-600 ring-2 ring-blue-200'
                      : ''
                  } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        {getEfficiencyIcon(method.efficiency)}
                        <CardTitle className="text-lg">{method.name}</CardTitle>
                      </div>
                      {getPerformanceBadge(method.performance)}
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {method.description}
                    </p>
                  </CardHeader>
                  <CardContent className="flex-1 flex flex-col justify-between">
                    <div className="flex flex-wrap gap-2 mb-3">
                      {method.badges.map((badge: string) => (
                        <Badge key={badge} variant="outline" className="text-xs">
                          {badge}
                        </Badge>
                      ))}
                    </div>
                    <div className="space-y-1 text-xs text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Target className="w-3 h-3" />
                        <Tooltip>
                          <TooltipTrigger className="underline decoration-dotted">
                            Param: {method.parameterReduction}
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Percentage of parameters reduced</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <div className="flex items-center gap-2">
                        <Layers className="w-3 h-3" />
                        <Tooltip>
                          <TooltipTrigger className="underline decoration-dotted">
                            Memory: {method.memoryReduction}
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Memory reduction during training</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <div className="flex items-center gap-2">
                        <GitBranch className="w-3 h-3" />
                        <span>Setup: {method.difficulty}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Label>
            </div>
          ))}
        </div>
      </RadioGroup>
    </TooltipProvider>
  );
}

export function getMethodById(id: TrainingMethod): TrainingMethodInfo | undefined {
  return trainingMethods.find((method: TrainingMethodInfo) => method.id === id);
}

export function getDefaultMethod(): TrainingMethod {
  return 'dora';
}
