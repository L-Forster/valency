'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function Hero() {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  // Subtle background animation
  const pulseVariants = {
    initial: { scale: 1, opacity: 0.1 },
    animate: {
      scale: 1.1,
      opacity: [0.1, 0.2, 0.1],
      transition: {
        duration: 8,
        repeat: Infinity,
        repeatType: "reverse" as const,
      },
    },
  };

  return (
    <div className="relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 z-0">
        <motion.div 
          className="absolute rounded-full w-[500px] h-[500px] bg-deepblue-900/20 blur-3xl"
          style={{ top: '-150px', right: '-100px' }}
          variants={pulseVariants}
          initial="initial"
          animate="animate"
        />
        <motion.div 
          className="absolute rounded-full w-[600px] h-[600px] bg-purple-900/20 blur-3xl"
          style={{ bottom: '-200px', left: '-150px' }}
          variants={pulseVariants}
          initial="initial"
          animate="animate"
          transition={{ delay: 2 }}
        />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 sm:pt-32 sm:pb-24 flex flex-col items-center justify-center text-center">
        <motion.div
          className="max-w-4xl mx-auto"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.h1
            variants={itemVariants}
            className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-8"
          >
            <span className="block text-gradient">Understand the emotional impact</span>
            <span className="block text-slate-100">of your chatbot conversations</span>
          </motion.h1>
          
          <motion.p
            variants={itemVariants}
            className="mt-6 text-xl text-slate-300 max-w-3xl mx-auto"
          >
            ToneAnalytics uses LSTM neural networks to analyze emotional valence and arousal in text, helping businesses create more empathetic chatbot experiences.
          </motion.p>
          
          <motion.div 
            variants={itemVariants}
            className="mt-10"
          >
            <Link 
              href="#demo"
              className="btn btn-primary text-lg px-8 py-3 shadow-glow-md"
            >
              Try Demo
            </Link>
          </motion.div>
          
          <motion.div 
            variants={itemVariants}
            className="mt-16 flex justify-center"
          >
            <div className="flex space-x-3 text-sm text-slate-400">
              <span className="flex items-center">
                <svg className="h-5 w-5 text-accent-teal mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Advanced Emotional AI
              </span>
              <span className="flex items-center">
                <svg className="h-5 w-5 text-accent-teal mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Real-time Analysis
              </span>
              <span className="flex items-center">
                <svg className="h-5 w-5 text-accent-teal mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Actionable Insights
              </span>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
} 