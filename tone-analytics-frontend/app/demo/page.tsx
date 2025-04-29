import React from 'react';
import DemoWidget from '../components/demo/DemoWidget';

export default function DemoPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Try Valency AI Demo
          </h1>
          <p className="mt-3 max-w-2xl mx-auto text-xl text-gray-500 sm:mt-4">
            Analyze the emotional tone of your chatbot conversations using our powerful transformer neural network.
          </p>
        </div>
        
        <div className="max-w-4xl mx-auto">
          <DemoWidget />
        </div>
        
        <div className="mt-16 max-w-2xl mx-auto text-center">
          <h2 className="text-2xl font-bold text-gray-900">
            Ready to improve your chatbot interactions?
          </h2>
          <p className="mt-4 text-lg text-gray-500">
            Sign up for a free trial to get full access to all features and analytics.
          </p>
          <div className="mt-8">
            <a
              href="/#pricing"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Start Free Trial
            </a>
          </div>
        </div>
      </div>
    </div>
  );
} 