// Global type declarations for JSX modules

declare module '*.jsx' {
  import * as React from 'react';
  const component: React.FC<any>;
  export default component;
}

// Specific UI component declarations
declare module '@/components/ui/card.jsx' {
  import * as React from 'react';
  
  interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}
  interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}
  interface CardTitleProps extends React.HTMLAttributes<HTMLDivElement> {}
  interface CardDescriptionProps extends React.HTMLAttributes<HTMLDivElement> {}
  interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {}
  interface CardFooterProps extends React.HTMLAttributes<HTMLDivElement> {}
  
  export const Card: React.ForwardRefExoticComponent<CardProps & React.RefAttributes<HTMLDivElement>>;
  export const CardHeader: React.ForwardRefExoticComponent<CardHeaderProps & React.RefAttributes<HTMLDivElement>>;
  export const CardTitle: React.ForwardRefExoticComponent<CardTitleProps & React.RefAttributes<HTMLDivElement>>;
  export const CardDescription: React.ForwardRefExoticComponent<CardDescriptionProps & React.RefAttributes<HTMLDivElement>>;
  export const CardContent: React.ForwardRefExoticComponent<CardContentProps & React.RefAttributes<HTMLDivElement>>;
  export const CardFooter: React.ForwardRefExoticComponent<CardFooterProps & React.RefAttributes<HTMLDivElement>>;
}

declare module '@/components/ui/radio-group.jsx' {
  import * as React from 'react';
  
  interface RadioGroupProps extends React.HTMLAttributes<HTMLDivElement> {
    value?: string;
    onValueChange?: (value: string) => void;
    disabled?: boolean;
  }
  
  interface RadioGroupItemProps extends React.HTMLAttributes<HTMLButtonElement> {
    value: string;
    id?: string;
    disabled?: boolean;
  }
  
  export const RadioGroup: React.ForwardRefExoticComponent<RadioGroupProps & React.RefAttributes<HTMLDivElement>>;
  export const RadioGroupItem: React.ForwardRefExoticComponent<RadioGroupItemProps & React.RefAttributes<HTMLButtonElement>>;
}

declare module '@/components/ui/tooltip.jsx' {
  import * as React from 'react';
  
  interface TooltipProps {
    children?: React.ReactNode;
  }
  
  interface TooltipContentProps extends React.HTMLAttributes<HTMLDivElement> {
    sideOffset?: number;
  }
  
  interface TooltipTriggerProps {
    children?: React.ReactNode;
    asChild?: boolean;
  }
  
  interface TooltipProviderProps {
    children?: React.ReactNode;
    delayDuration?: number;
  }
  
  export const Tooltip: React.FC<TooltipProps>;
  export const TooltipContent: React.ForwardRefExoticComponent<TooltipContentProps & React.RefAttributes<HTMLDivElement>>;
  export const TooltipTrigger: React.FC<TooltipTriggerProps>;
  export const TooltipProvider: React.FC<TooltipProviderProps>;
}

declare module '@/components/ui/badge.jsx' {
  import * as React from 'react';
  
  interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'secondary' | 'outline';
  }
  
  export const Badge: React.FC<BadgeProps>;
}

declare module '@/components/ui/label.jsx' {
  import * as React from 'react';
  
  interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {}
  
  export const Label: React.FC<LabelProps>;
}

// Relative path declarations
declare module '../ui/card.jsx' {
  export * from '@/components/ui/card.jsx';
}

declare module '../ui/radio-group.jsx' {
  export * from '@/components/ui/radio-group.jsx';
}

declare module '../ui/tooltip.jsx' {
  export * from '@/components/ui/tooltip.jsx';
}

declare module '../ui/badge.jsx' {
  export * from '@/components/ui/badge.jsx';
}

declare module '../ui/label.jsx' {
  export * from '@/components/ui/label.jsx';
}

declare module '../ui/input.jsx' {
  export * from '@/components/ui/input.jsx';
}

declare module '../ui/slider.jsx' {
  export * from '@/components/ui/slider.jsx';
}

declare module '../ui/switch.jsx' {
  export * from '@/components/ui/switch.jsx';
}

declare module '../ui/alert.jsx' {
  export * from '@/components/ui/alert.jsx';
}
