import React from 'react';
import Hero from './components/home/Hero';
import Features from './components/home/Features';
import DemoWidget from './components/demo/DemoWidget';
import Contact from './components/home/Contact';

export default function Home() {
  return (
    <main className="flex flex-col">
      {/* Hero Section */}
      <section id="hero" className="section">
        <Hero />
      </section>
      
      {/* Demo Widget */}
      <section id="demo" className="section bg-dark-300">
        <DemoWidget />
      </section>
      
      {/* Features Section */}
      <section id="features" className="section">
        <Features />
      </section>
      
      {/* Contact Section */}
      <section id="contact" className="section bg-dark-300">
        <Contact />
      </section>
    </main>
  );
}
