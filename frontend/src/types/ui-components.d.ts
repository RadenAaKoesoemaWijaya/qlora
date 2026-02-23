// Type declarations for UI components
declare module '../ui/card.jsx' {
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