// Type declarations for UI components - Fixed paths
declare module '*/ui/card.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, HTMLAttributes } from 'react';
  
  export interface CardProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export interface CardTitleProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export interface CardContentProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export const Card: ForwardRefExoticComponent<CardProps & RefAttributes<HTMLDivElement>>;
  export const CardHeader: ForwardRefExoticComponent<CardHeaderProps & RefAttributes<HTMLDivElement>>;
  export const CardTitle: ForwardRefExoticComponent<CardTitleProps & RefAttributes<HTMLDivElement>>;
  export const CardContent: ForwardRefExoticComponent<CardContentProps & RefAttributes<HTMLDivElement>>;
}

declare module '*/ui/button.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, ButtonHTMLAttributes } from 'react';
  
  export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    className?: string;
    variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
    size?: 'default' | 'sm' | 'lg';
    asChild?: boolean;
  }
  
  export const Button: ForwardRefExoticComponent<ButtonProps & RefAttributes<HTMLButtonElement>>;
}

declare module '*/ui/input.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, InputHTMLAttributes } from 'react';
  
  export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    className?: string;
  }
  
  export const Input: ForwardRefExoticComponent<InputProps & RefAttributes<HTMLInputElement>>;
}

declare module '*/ui/label.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, LabelHTMLAttributes } from 'react';
  
  export interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
    className?: string;
  }
  
  export const Label: ForwardRefExoticComponent<LabelProps & RefAttributes<HTMLLabelElement>>;
}

declare module '*/ui/select.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, ButtonHTMLAttributes, HTMLAttributes } from 'react';
  
  export interface SelectProps {
    children: React.ReactNode;
    value?: string;
    onValueChange?: (value: string) => void;
    defaultValue?: string;
    disabled?: boolean;
  }
  
  export interface SelectTriggerProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    className?: string;
    children: React.ReactNode;
  }
  
  export interface SelectContentProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    children: React.ReactNode;
    position?: 'popper' | 'item-aligned';
  }
  
  export interface SelectItemProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    children: React.ReactNode;
    value: string;
    disabled?: boolean;
  }
  
  export interface SelectValueProps {
    placeholder?: string;
    children?: React.ReactNode;
  }
  
  export const Select: React.FC<SelectProps>;
  export const SelectTrigger: ForwardRefExoticComponent<SelectTriggerProps & RefAttributes<HTMLButtonElement>>;
  export const SelectContent: React.FC<SelectContentProps>;
  export const SelectItem: ForwardRefExoticComponent<SelectItemProps & RefAttributes<HTMLDivElement>>;
  export const SelectValue: React.FC<SelectValueProps>;
}

declare module '*/ui/slider.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, HTMLAttributes } from 'react';
  
  export interface SliderProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    defaultValue?: number[];
    value?: number[];
    onValueChange?: (value: number[]) => void;
    max?: number;
    min?: number;
    step?: number;
    disabled?: boolean;
  }
  
  export const Slider: ForwardRefExoticComponent<SliderProps & RefAttributes<HTMLDivElement>>;
}

declare module '*/ui/switch.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, ButtonHTMLAttributes } from 'react';
  
  export interface SwitchProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    className?: string;
    checked?: boolean;
    onCheckedChange?: (checked: boolean) => false;
    disabled?: boolean;
  }
  
  export const Switch: ForwardRefExoticComponent<SwitchProps & RefAttributes<HTMLButtonElement>>;
}

declare module '*/ui/textarea.jsx' {
  import { ForwardRefExoticComponent, RefAttributes, TextareaHTMLAttributes } from 'react';
  
  export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
    className?: string;
  }
  
  export const Textarea: ForwardRefExoticComponent<TextareaProps & RefAttributes<HTMLTextAreaElement>>;
}

declare module '*/ui/badge.jsx' {
  import { HTMLAttributes } from 'react';
  
  export interface BadgeProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    variant?: 'default' | 'secondary' | 'destructive' | 'outline';
  }
  
  export const Badge: React.FC<BadgeProps>;
}

declare module '*/ui/alert.jsx' {
  import { HTMLAttributes } from 'react';
  
  export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export interface AlertTitleProps extends HTMLAttributes<HTMLHeadingElement> {
    className?: string;
  }
  
  export interface AlertDescriptionProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export const Alert: React.FC<AlertProps>;
  export const AlertTitle: React.FC<AlertTitleProps>;
  export const AlertDescription: React.FC<AlertDescriptionProps>;
}

declare module '*/ui/tabs.jsx' {
  import { HTMLAttributes, ButtonHTMLAttributes } from 'react';
  
  export interface TabsProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    defaultValue?: string;
    value?: string;
    onValueChange?: (value: string) => void;
  }
  
  export interface TabsListProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
  }
  
  export interface TabsTriggerProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    className?: string;
    value: string;
    disabled?: boolean;
  }
  
  export interface TabsContentProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    value: string;
  }
  
  export const Tabs: React.FC<TabsProps>;
  export const TabsList: React.FC<TabsListProps>;
  export const TabsTrigger: React.FC<TabsTriggerProps>;
  export const TabsContent: React.FC<TabsContentProps>;
}

declare module '*/ui/collapsible.jsx' {
  import { HTMLAttributes } from 'react';
  
  export interface CollapsibleProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
    disabled?: boolean;
  }
  
  export interface CollapsibleTriggerProps extends HTMLAttributes<HTMLButtonElement> {
    className?: string;
    children: React.ReactNode;
    asChild?: boolean;
  }
  
  export interface CollapsibleContentProps extends HTMLAttributes<HTMLDivElement> {
    className?: string;
    children: React.ReactNode;
  }
  
  export const Collapsible: React.FC<CollapsibleProps>;
  export const CollapsibleTrigger: React.FC<CollapsibleTriggerProps>;
  export const CollapsibleContent: React.FC<CollapsibleContentProps>;
}