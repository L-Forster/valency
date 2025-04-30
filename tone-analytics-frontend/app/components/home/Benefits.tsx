'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { ChatBubbleLeftRightIcon, ChartBarIcon, ArrowTrendingUpIcon, SparklesIcon } from '@heroicons/react/24/outline';

const benefits = [
  {
    name: 'Increased Customer Satisfaction',
    description:
      'Improve customer experience by ensuring your chatbots respond with emotionally appropriate tone, leading to higher satisfaction rates and better customer retention.',
    icon: ChatBubbleLeftRightIcon,
    color: 'bg-indigo-500',
  },
  {
    name: 'Reduced Customer Churn',
    description:
      'Identify and address negative emotional patterns in conversations before they result in customer loss. Our analytics help you pinpoint critical moments where interventions are needed.',
    icon: ArrowTrendingUpIcon,
    color: 'bg-pink-500',
  },
  {
    name: 'Data-Driven Conversation Design',
    description:
      'Use valence-arousal data to design more effective conversation flows. Our dashboards provide actionable insights to optimize your chatbot interactions.',
    icon: ChartBarIcon,
    color: 'bg-blue-500',
  },
  {
    name: 'AI-Powered Recommendations',
    description:
      'Receive automatic improvement suggestions based on your specific conversation patterns. Our transformer neural network learns from your data to provide customized recommendations.',
    icon: SparklesIcon,
    color: 'bg-purple-500',
  },
];

export default function Benefits() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 },
    },
  };

  return (
    <div className="bg-gray-50 py-16 sm:py-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:text-center mb-12">
          <h2 className="text-base text-indigo-600 font-semibold tracking-wide uppercase">Benefits</h2>
          <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
            The Business Impact of Emotional Intelligence
          </p>
          <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
            Our transformer-based valence-arousal regression model enables businesses to enhance chatbot interactions in ways that directly improve key performance metrics.
          </p>
        </div>

        <motion.div
          className="mt-10"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
            {benefits.map((benefit, index) => (
              <motion.div key={benefit.name} variants={itemVariants} className="relative">
                <dt>
                  <div className={`absolute flex items-center justify-center h-12 w-12 rounded-md ${benefit.color} text-white`}>
                    <benefit.icon className="h-6 w-6" aria-hidden="true" />
                  </div>
                  <p className="ml-16 text-lg leading-6 font-medium text-gray-900">{benefit.name}</p>
                </dt>
                <dd className="mt-2 ml-16 text-base text-gray-500">{benefit.description}</dd>
              </motion.div>
            ))}
          </dl>
        </motion.div>
      </div>
    </div>
  );
} 