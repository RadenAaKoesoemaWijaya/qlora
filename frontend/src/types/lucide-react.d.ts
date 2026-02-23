// Type declarations for lucide-react
declare module 'lucide-react' {
  import { FC, SVGProps } from 'react';
  
  export interface LucideProps extends SVGProps<SVGSVGElement> {
    size?: number;
    color?: string;
    strokeWidth?: number;
    absoluteStrokeWidth?: boolean;
  }
  
  export type LucideIcon = FC<LucideProps>;
  
  export const ChevronDown: LucideIcon;
  export const ChevronUp: LucideIcon;
  export const Info: LucideIcon;
  export const AlertTriangle: LucideIcon;
  export const CheckCircle: LucideIcon;
  export const Settings: LucideIcon;
  export const Cpu: LucideIcon;
  export const Database: LucideIcon;
  export const Zap: LucideIcon;
  export const Shield: LucideIcon;
  export const Clock: LucideIcon;
  export const Save: LucideIcon;
  export const RotateCcw: LucideIcon;
}