'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Button from '../ui/Button';

export default function CTA() {
  return (
    <div className="bg-indigo-700">
      <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
        <motion.h2
          className="text-3xl font-extrabold text-white sm:text-4xl"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <span className="block">Ready to transform your chatbot experience?</span>
          <span className="block">Start your free trial today.</span>
        </motion.h2>
        <motion.p
          className="mt-4 text-lg leading-6 text-indigo-200"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Enhance your customer service with emotional intelligence.
          <br />
          No credit card required for 14-day trial. Cancel anytime.
        </motion.p>
        <motion.div
          className="mt-8 flex justify-center gap-x-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Button 
            className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
          >
            Start free trial
          </Button>
          <Button 
            className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-800 hover:bg-indigo-900"
          >
            Learn more
          </Button>
        </motion.div>
        <motion.p
          className="mt-6 text-sm text-indigo-200"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          Join over 2,000 companies already improving their customer conversations
        </motion.p>
      </div>
    </div>
  );
} 