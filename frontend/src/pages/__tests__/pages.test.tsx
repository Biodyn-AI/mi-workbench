import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// Mock fetch for all page tests
const mockFetch = vi.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve([]),
    text: () => Promise.resolve(''),
  }),
);
globalThis.fetch = mockFetch as unknown as typeof fetch;

// Import pages
import Overview from '../Overview';
import Runs from '../Runs';
import Telemetry from '../Telemetry';
import Knowledge from '../Knowledge';
import Comparison from '../Comparison';

function renderWithRouter(element: React.ReactElement) {
  return render(<MemoryRouter>{element}</MemoryRouter>);
}

describe('Page Smoke Tests', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('Overview renders without crashing', () => {
    renderWithRouter(<Overview />);
    expect(document.body).toBeTruthy();
  });

  it('Runs renders without crashing', () => {
    renderWithRouter(<Runs />);
    expect(document.body).toBeTruthy();
  });

  it('Telemetry renders loading state', () => {
    renderWithRouter(<Telemetry />);
    expect(screen.getByText(/loading telemetry/i)).toBeTruthy();
  });

  it('Knowledge renders loading state', () => {
    renderWithRouter(<Knowledge />);
    expect(screen.getByText(/loading knowledge/i)).toBeTruthy();
  });

  it('Comparison renders loading state', () => {
    renderWithRouter(<Comparison />);
    expect(screen.getByText(/loading runs/i)).toBeTruthy();
  });
});
