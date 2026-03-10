// Type declarations for UI components
// This file provides TypeScript type declarations for JavaScript UI components

declare module '../ui/card.jsx' {
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

declare module '../ui/radio-group.jsx' {
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

declare module '../ui/tooltip.jsx' {
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

declare module '../ui/badge.jsx' {
  import * as React from 'react';
  
  interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'secondary' | 'outline';
  }
  
  export const Badge: React.FC<BadgeProps>;
}

declare module '../ui/label.jsx' {
  import * as React from 'react';
  
  interface LabelProps extends React.LabelHTMLAttributes<HTMLLabelElement> {}
  
  export const Label: React.FC<LabelProps>;
}

declare module '../ui/input.jsx' {
  import * as React from 'react';
  
  interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}
  
  export const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<HTMLInputElement>>;
}

declare module '../ui/slider.jsx' {
  import * as React from 'react';
  
  interface SliderProps {
    value?: number[];
    onValueChange?: (value: number[]) => void;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
  }
  
  export const Slider: React.FC<SliderProps>;
}

declare module '../ui/switch.jsx' {
  import * as React from 'react';
  
  interface SwitchProps {
    checked?: boolean;
    onCheckedChange?: (checked: boolean) => void;
    disabled?: boolean;
    id?: string;
  }
  
  export const Switch: React.FC<SwitchProps>;
}

declare module '../ui/alert.jsx' {
  import * as React from 'react';
  
  interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: 'default' | 'destructive';
  }
  
  interface AlertTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {}
  interface AlertDescriptionProps extends React.HTMLAttributes<HTMLParagraphElement> {}
  
  export const Alert: React.FC<AlertProps>;
  export const AlertTitle: React.FC<AlertTitleProps>;
  export const AlertDescription: React.FC<AlertDescriptionProps>;
}
