'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import AnalysisInput from './AnalysisInput';
import AnalysisResults from './AnalysisResults';
import { EnhancedAnalysisResult } from '../../types';

export default function DemoWidget() {
  const [result, setResult] = useState<EnhancedAnalysisResult | null>(null);
  const [originalText, setOriginalText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleAnalysisComplete = (analysisResult: EnhancedAnalysisResult) => {
    setResult(analysisResult);
    const textElement = document.getElementById('conversation-text') as HTMLTextAreaElement;
    setOriginalText(textElement?.value || '');
  };

  const handleReset = () => {
    setResult(null);
    setOriginalText('');
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      <motion.div 
        className="text-center mb-10"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
      >
        <h2 className="text-3xl font-bold text-gradient mb-4">
          Experience the Analysis
        </h2>
        <p className="max-w-2xl mx-auto text-lg text-slate-300">
          Paste a chatbot conversation and see how our transformer neural network detects emotional patterns.
        </p>
      </motion.div>

      <motion.div
        className="card border border-dark-100 overflow-hidden backdrop-blur-sm bg-dark-100/80"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        viewport={{ once: true }}
      >
        {!result ? (
          <AnalysisInput 
            onAnalysisComplete={handleAnalysisComplete} 
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex justify-end mb-4">
              <button
                onClick={handleReset}
                className="btn btn-secondary text-sm px-4 py-2"
              >
                <svg 
                  className="h-4 w-4 mr-2" 
                  xmlns="http://www.w3.org/2000/svg" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Analyze Another Text
              </button>
            </div>
            <AnalysisResults result={result} originalText={originalText} />
          </motion.div>
        )}
      </motion.div>
      
      {/* Sample conversation prompt */}
      {!result && (
        <motion.div 
          className="mt-8 text-center text-sm text-slate-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          <p className="mb-2 font-medium">Not sure what to try? Use this sample conversation:</p>
          <div className="inline-block text-left bg-dark-300 p-4 rounded-md">
            <p>Customer: I&apos;ve been waiting for over 30 minutes to connect with a representative!</p>
            <p>Chatbot: I&apos;m sorry for the wait time. I&apos;ll help you right away.</p>
            <p>Customer: This is ridiculous. I just want to update my shipping address.</p>
            <p>Chatbot: I understand your frustration. I can update your shipping address now.</p>
            <p>Customer: Thanks, I appreciate that. My new address is 123 Main St.</p>
            <p>Chatbot: Your shipping address has been updated successfully.</p>
          </div>
        </motion.div>
      )}
    </div>
  );
} 