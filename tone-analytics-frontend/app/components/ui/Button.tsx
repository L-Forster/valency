'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

type ButtonVariant = 'primary' | 'secondary' | 'tertiary' | 'ghost';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps {
  children: React.ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  href?: string;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  className?: string;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  isLoading?: boolean;
}

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  href,
  type = 'button',
  disabled = false,
  fullWidth = false,
  onClick,
  className = '',
  icon,
  iconPosition = 'left',
  isLoading = false,
}: ButtonProps) {
  // Base classes for all buttons
  const baseClasses = 'font-semibold rounded-lg inline-flex items-center justify-center transition-all';
  
  // Size specific classes
  const sizeClasses = {
    sm: 'text-sm py-1.5 px-3',
    md: 'text-base py-2 px-4',
    lg: 'text-lg py-3 px-6',
  };
  
  // Variant specific classes
  const variantClasses = {
    primary: 'bg-indigo-600 text-white hover:bg-indigo-700 active:bg-indigo-800 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:outline-none',
    secondary: 'bg-white text-indigo-600 border border-indigo-600 hover:bg-indigo-50 active:bg-indigo-100 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:outline-none',
    tertiary: 'bg-gray-100 text-gray-800 hover:bg-gray-200 active:bg-gray-300 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:outline-none',
    ghost: 'bg-transparent text-indigo-600 hover:bg-indigo-50 active:bg-indigo-100 focus:outline-none',
  };
  
  // Disabled classes
  const disabledClasses = 'opacity-60 cursor-not-allowed';
  
  // Full width class
  const fullWidthClass = fullWidth ? 'w-full' : '';
  
  // Loading classes
  const loadingClasses = isLoading ? 'cursor-wait' : '';
  
  // Combine all classes
  const buttonClasses = `
    ${baseClasses}
    ${sizeClasses[size]}
    ${variantClasses[variant]}
    ${disabled || isLoading ? disabledClasses : ''}
    ${fullWidthClass}
    ${loadingClasses}
    ${className}
  `;
  
  // Framer motion variants for hover animation
  const buttonVariants = {
    initial: { scale: 1 },
    hover: { scale: 1.02 },
    tap: { scale: 0.98 },
  };
  
  // If href is provided, render as Link
  if (href) {
    return (
      <Link href={href} passHref>
        <motion.a
          className={buttonClasses}
          initial="initial"
          whileHover={!disabled && !isLoading ? "hover" : "initial"}
          whileTap={!disabled && !isLoading ? "tap" : "initial"}
          variants={buttonVariants}
        >
          {isLoading && (
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          )}
          {icon && iconPosition === 'left' && !isLoading && <span className="mr-2">{icon}</span>}
          {children}
          {icon && iconPosition === 'right' && <span className="ml-2">{icon}</span>}
        </motion.a>
      </Link>
    );
  }
  
  // Otherwise, render as button
  return (
    <motion.button
      type={type}
      disabled={disabled || isLoading}
      className={buttonClasses}
      onClick={onClick}
      initial="initial"
      whileHover={!disabled && !isLoading ? "hover" : "initial"}
      whileTap={!disabled && !isLoading ? "tap" : "initial"}
      variants={buttonVariants}
    >
      {isLoading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {icon && iconPosition === 'left' && !isLoading && <span className="mr-2">{icon}</span>}
      {children}
      {icon && iconPosition === 'right' && <span className="ml-2">{icon}</span>}
    </motion.button>
  );
} 