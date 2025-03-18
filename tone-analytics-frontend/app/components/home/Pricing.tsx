'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckIcon } from '@heroicons/react/24/solid';
import Button from '../ui/Button';

type Frequency = 'monthly' | 'annually';

interface PricingTier {
  name: string;
  id: string;
  price: {
    monthly: number;
    annually: number;
  };
  description: string;
  features: string[];
  mostPopular: boolean;
}

const tiers: PricingTier[] = [
  {
    name: 'Starter',
    id: 'tier-starter',
    price: { monthly: 49, annually: 39 },
    description: 'Perfect for startups and small businesses just getting started with chatbots.',
    features: [
      'Basic emotion analytics',
      'Up to 10,000 conversations/month',
      'Weekly reports',
      'Email support',
      '1 chatbot integration',
    ],
    mostPopular: false,
  },
  {
    name: 'Professional',
    id: 'tier-professional',
    price: { monthly: 99, annually: 79 },
    description: 'Ideal for growing businesses with established chatbot systems.',
    features: [
      'Advanced emotion analytics',
      'Up to 50,000 conversations/month',
      'Daily reports',
      'Priority email support',
      '5 chatbot integrations',
      'API access',
      'Custom dashboards',
    ],
    mostPopular: true,
  },
  {
    name: 'Enterprise',
    id: 'tier-enterprise',
    price: { monthly: 249, annually: 199 },
    description: 'For large organizations with complex chatbot ecosystems.',
    features: [
      'Complete emotion analytics suite',
      'Unlimited conversations',
      'Real-time reporting',
      '24/7 dedicated support',
      'Unlimited chatbot integrations',
      'Advanced API access',
      'Custom model training',
      'Dedicated account manager',
      'SLA guarantees',
    ],
    mostPopular: false,
  },
];

export default function Pricing() {
  const [frequency, setFrequency] = useState<Frequency>('monthly');

  return (
    <div id="pricing" className="bg-gray-50 py-16 sm:py-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-base font-semibold tracking-wide uppercase text-indigo-600">Pricing</h2>
          <p className="mt-2 text-3xl font-extrabold text-gray-900 sm:text-4xl lg:text-5xl">
            The right price for your needs
          </p>
          <p className="mt-4 max-w-2xl text-xl text-gray-500 mx-auto">
            Choose the plan that works best for your business. All plans include a 14-day free trial.
          </p>
        </div>

        <div className="mt-12 sm:mt-16 sm:flex sm:justify-center">
          <div className="relative bg-white rounded-lg p-0.5 flex sm:max-w-md">
            <button
              type="button"
              className={`relative w-1/2 py-2 text-sm font-medium text-indigo-700 rounded-md whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:z-10 sm:w-auto sm:px-8 ${
                frequency === 'monthly' ? 'bg-white shadow-sm' : 'text-gray-700'
              }`}
              onClick={() => setFrequency('monthly')}
            >
              Monthly
            </button>
            <button
              type="button"
              className={`relative w-1/2 py-2 text-sm font-medium text-indigo-700 rounded-md whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:z-10 sm:w-auto sm:px-8 ${
                frequency === 'annually' ? 'bg-white shadow-sm' : 'text-gray-700'
              }`}
              onClick={() => setFrequency('annually')}
            >
              Annually <span className="text-indigo-500 text-xs">Save 20%</span>
            </button>
          </div>
        </div>

        <div className="mt-12 space-y-12 lg:space-y-0 lg:grid lg:grid-cols-3 lg:gap-8">
          {tiers.map((tier) => (
            <motion.div
              key={tier.id}
              className={`relative p-8 bg-white border rounded-2xl shadow-sm flex flex-col ${
                tier.mostPopular ? 'ring-2 ring-indigo-600' : 'border-gray-200'
              }`}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.5 }}
            >
              {tier.mostPopular && (
                <div className="absolute top-0 right-6 -mt-3 px-4 py-1 bg-indigo-600 rounded-full text-xs font-semibold uppercase tracking-wide text-white">
                  Most popular
                </div>
              )}
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900">{tier.name}</h3>
                <p className="mt-4 flex items-baseline text-gray-900">
                  <span className="text-5xl font-extrabold tracking-tight">${tier.price[frequency]}</span>
                  <span className="ml-1 text-xl font-semibold">/month</span>
                </p>
                <p className="mt-6 text-gray-500">{tier.description}</p>

                <ul className="mt-6 space-y-4">
                  {tier.features.map((feature) => (
                    <li key={feature} className="flex">
                      <CheckIcon className="flex-shrink-0 h-6 w-6 text-green-500" aria-hidden="true" />
                      <span className="ml-3 text-gray-500">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <Button
                className={`mt-8 block w-full ${
                  tier.mostPopular ? 'bg-indigo-600 hover:bg-indigo-700 text-white' : 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100'
                } py-3 px-6 border border-transparent rounded-md shadow text-center font-medium`}
              >
                {tier.mostPopular ? 'Start your trial' : 'Start now'}
              </Button>
            </motion.div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <p className="text-gray-600">
            Need a custom plan for your enterprise?{' '}
            <a href="#contact" className="font-medium text-indigo-600 hover:text-indigo-500">
              Contact us
            </a>
          </p>
        </div>
      </div>
    </div>
  );
} 