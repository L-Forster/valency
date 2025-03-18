'use client';

import React from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';

interface Testimonial {
  content: string;
  author: {
    name: string;
    title: string;
    company: string;
    image: string;
  };
}

const testimonials: Testimonial[] = [
  {
    content:
      "Valency AI has transformed our customer service chatbot. We've seen a 23% increase in customer satisfaction scores and a significant decrease in escalated issues since implementing their emotional analysis system.",
    author: {
      name: 'Sarah Johnson',
      title: 'Head of Customer Experience',
      company: 'TechSolutions Inc.',
      image: '/images/testimonials/sarah.jpg',
    },
  },
  {
    content:
      "The level of detail in the emotional analytics has been eye-opening. We discovered patterns in our chatbot conversations that were causing customer frustration that we never would have identified without Valency AI's dimensional approach.",
    author: {
      name: 'Michael Chen',
      title: 'AI Implementation Lead',
      company: 'Global Retail Group',
      image: '/images/testimonials/michael.jpg',
    },
  },
  {
    content:
      "What sets Valency AI apart is how actionable their insights are. The specific recommendations for improving our conversation flows have been invaluable, and we've been able to implement changes that had an immediate positive impact.",
    author: {
      name: 'Priya Patel',
      title: 'Director of Digital Strategy',
      company: 'FinServe Solutions',
      image: '/images/testimonials/priya.jpg',
    },
  },
];

export default function Testimonials() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.6 },
    },
  };

  return (
    <div className="bg-white py-16 sm:py-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-base font-semibold tracking-wide uppercase text-indigo-600">Testimonials</h2>
          <p className="mt-2 text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Trusted by leading companies
          </p>
          <p className="mt-4 max-w-2xl text-xl text-gray-500 mx-auto">
            See how businesses are using Valency AI to improve their chatbot conversations
          </p>
        </div>

        <motion.div 
          className="mt-16 grid gap-8 lg:grid-cols-3 lg:gap-x-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              className="relative bg-white rounded-2xl shadow-lg p-8 border border-gray-200"
              variants={itemVariants}
            >
              <div className="relative h-12 w-12 mx-auto mb-6">
                <div className="h-12 w-12 rounded-full bg-indigo-100 flex items-center justify-center text-gray-400">
                  {/* Placeholder for testimonial avatar - replace with actual images */}
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                  </svg>
                </div>
              </div>
              <blockquote>
                <p className="text-lg font-medium text-gray-900 text-center mb-8">&quot;{testimonial.content}&quot;</p>
                <div className="flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-base font-semibold text-gray-900">{testimonial.author.name}</div>
                    <div className="text-sm text-gray-500">{testimonial.author.title}</div>
                    <div className="text-sm font-medium text-indigo-600">{testimonial.author.company}</div>
                  </div>
                </div>
              </blockquote>
            </motion.div>
          ))}
        </motion.div>

        <div className="mt-16 border-t border-gray-200 pt-12">
          <div className="text-center">
            <p className="text-base font-semibold tracking-wider text-gray-500 uppercase">
              Trusted by innovative companies worldwide
            </p>
            <div className="mt-6 grid grid-cols-2 gap-8 md:grid-cols-6 lg:grid-cols-5">
              {/* Company logos - replace with actual logos */}
              {[...Array(5)].map((_, i) => (
                <div key={i} className="col-span-1 flex justify-center md:col-span-2 lg:col-span-1">
                  <div className="h-12 w-full bg-gray-100 rounded flex items-center justify-center text-gray-400">
                    <span className="text-xs">Company Logo</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}