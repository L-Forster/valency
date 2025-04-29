'use client';

import React from 'react';
import { motion } from 'framer-motion';

const features = [
  {
    name: 'Emotional Valence & Arousal Analysis',
    description: 'State-of-the-art transformer neural networks quantify emotional dimensions in text for deeper conversation understanding.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    name: 'Real-time Conversation Analysis',
    description: 'Instantly analyze chat interactions to provide moment-by-moment emotional insights for immediate action.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  {
    name: 'Actionable Improvement Insights',
    description: 'Convert emotional data into concrete suggestions to enhance chatbot responses and customer satisfaction.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    name: 'Seamless Integration',
    description: 'Connect to your existing chatbot infrastructure with minimal setup through our simple API or pre-built connectors.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
];

export default function Features() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
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

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <motion.div 
        className="text-center mb-16"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-bold text-gradient mb-4">Key Features</h2>
        <p className="max-w-2xl mx-auto text-lg text-slate-300">
          Powerful tools to enhance your chatbot&apos;s emotional intelligence.
        </p>
      </motion.div>

      <motion.div 
        className="grid gap-12 md:grid-cols-2 lg:gap-16"
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        {features.map((feature, index) => (
          <motion.div 
            key={index} 
            className="card hover:border-purple-600 transition-all duration-300 flex flex-col h-full"
            variants={itemVariants}
          >
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0 bg-gradient-to-r from-deepblue-600 to-purple-600 rounded-md p-2 text-white">
                {feature.icon}
              </div>
              <h3 className="ml-4 text-xl font-medium text-slate-100">{feature.name}</h3>
            </div>
            <p className="text-slate-300">{feature.description}</p>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
} 