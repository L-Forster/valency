'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { analyzeText, checkApiAvailability } from '../../services/api';
import { EnhancedAnalysisResult } from '../../types';

interface AnalysisInputProps {
  onAnalysisComplete: (result: EnhancedAnalysisResult) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export default function AnalysisInput({ onAnalysisComplete, isLoading, setIsLoading }: AnalysisInputProps) {
  const [text, setText] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<string>('Checking API availability...');
  
  // Try to connect to the backend API first, fallback to demo mode if unavailable
  useEffect(() => {
    const checkApi = async () => {
      setApiStatus('Checking API availability...');
      const isAvailable = await checkApiAvailability();
      setApiStatus(isAvailable 
        ? 'Connected to neural network API' 
        : 'Using demo mode (neural network API not available)');
    };
    
    checkApi();
  }, []);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    if (error) setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }
    
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await analyzeText(text);
      onAnalysisComplete(result);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze text. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseExample = () => {
    const examples = [
      "Customer: I've been waiting for over 30 minutes to connect with a representative!\nChatbot: I'm sorry for the wait time. I'll help you right away.\nCustomer: This is ridiculous. I just want to update my shipping address.\nChatbot: I understand your frustration. I can update your shipping address now.\nCustomer: Thanks, I appreciate that. My new address is 123 Main St.\nChatbot: Your shipping address has been updated successfully.",
      "Customer: Hello, I need help setting up my new device. I've been trying for hours and nothing works.\nChatbot: I'm sorry to hear you're having trouble. What device are you trying to set up?\nCustomer: It's the XR200 smart home hub. The instructions are so confusing.\nChatbot: I understand that can be frustrating. Let me guide you through the setup step by step.\nCustomer: Thank you, that would be very helpful.\nChatbot: Great! First, make sure the device is powered on. Do you see a blue light on the front?",
      "Customer: I want to cancel my subscription immediately. I'm being charged for services I don't use.\nChatbot: I'd be happy to help you with cancellation. Could you please confirm your account email?\nCustomer: It's johndoe@example.com. I've been charged twice this month!\nChatbot: I apologize for the inconvenience. I can see the duplicate charge and will process a refund right away.\nCustomer: Oh, that's good to hear. How long will the refund take?\nChatbot: The refund should appear in your account within 3-5 business days. Is there anything else I can help with today?",
    ];
    
    const selectedExample = examples[Math.floor(Math.random() * examples.length)];
    setText(selectedExample);
  };

  return (
    <motion.div
      className="w-full max-w-3xl mx-auto bg-dark-100 rounded-xl shadow-lg p-6 mb-8 border border-deepblue-700/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-2xl font-bold mb-4 text-gradient">Text Analysis</h2>
      
      <div className="mb-4 text-sm text-slate-400 flex items-center">
        <span className={`inline-block h-2 w-2 rounded-full mr-2 ${apiStatus.includes('demo') ? 'bg-amber-500' : 'bg-emerald-500'}`}></span>
        {apiStatus}
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="text-input" className="block mb-2 text-sm font-medium text-slate-300">
            Enter conversation text to analyze
          </label>
          <textarea
            id="text-input"
            rows={5}
            className="input w-full"
            placeholder="Enter a customer service conversation, support chat, or any text you'd like to analyze for emotional tone..."
            value={text}
            onChange={handleTextChange}
            disabled={isLoading}
          />
          {error && (
            <motion.p 
              className="mt-2 text-red-400 text-sm"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {error}
            </motion.p>
          )}
        </div>
        
        <div className="flex justify-end">
          <button
            type="submit"
            className="btn btn-primary px-6"
            disabled={isLoading}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Text'}
          </button>
        </div>
        
        <p className="mt-4 text-xs text-slate-500">
          Try analyzing conversations to identify emotional patterns, tone changes, 
          and potential improvements to enhance customer interactions.
        </p>
      </form>
    </motion.div>
  );
} 