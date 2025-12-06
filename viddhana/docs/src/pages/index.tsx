import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: string;
  link: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Atlas Chain',
    icon: '⛓️',
    description: 'Layer 3 AppChain with 100,000+ TPS, 2-second blocks, and micro-transaction costs.',
    link: '/docs/getting-started/quickstart',
  },
  {
    title: 'Smart Contracts',
    icon: '📜',
    description: 'Audited, upgradeable contracts for policy engine, vaults, and risk control.',
    link: '/docs/smart-contracts/overview',
  },
  {
    title: 'JSON-RPC API',
    icon: '🔌',
    description: 'Complete API reference for interacting with the VIDDHANA network.',
    link: '/docs/api-reference/overview',
  },
  {
    title: 'SDKs',
    icon: '🛠️',
    description: 'Official JavaScript and Python SDKs for rapid development.',
    link: '/docs/sdks/javascript',
  },
];

function Feature({ title, icon, description, link }: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <Link to={link} className="feature-card">
        <div className="feature-card__icon">{icon}</div>
        <Heading as="h3" className="feature-card__title">{title}</Heading>
        <p className="feature-card__description">{description}</p>
      </Link>
    </div>
  );
}

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            Get Started
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/api-reference/overview"
            style={{ marginLeft: '1rem', color: 'white' }}>
            API Reference
          </Link>
        </div>
      </div>
    </header>
  );
}

function HomepageFeatures(): React.ReactElement {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.ReactElement {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Build on VIDDHANA: The Operating System for Wealth. Developer documentation, API reference, and SDKs.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
