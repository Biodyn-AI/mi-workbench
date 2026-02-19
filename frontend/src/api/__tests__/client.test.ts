import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch as typeof fetch;

import {
  listWorkspaces,
  listRuns,
  getRun,
  createRun,
  getKnowledgeSummary,
  compareRuns,
  type RunState,
} from '../client';

describe('API Client', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('listWorkspaces calls correct endpoint', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve([]),
    });
    const result = await listWorkspaces();
    expect(mockFetch).toHaveBeenCalledWith('/api/workspaces', expect.objectContaining({
      headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
    }));
    expect(result).toEqual([]);
  });

  it('listRuns includes workspace filter', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve([]),
    });
    await listRuns('ws123');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('workspace_id=ws123'),
      expect.anything(),
    );
  });

  it('createRun sends POST with body', async () => {
    const mockRun: Partial<RunState> = {
      run_id: 'test123',
      status: 'pending',
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockRun),
    });
    await createRun({
      workspace_id: 'ws1',
      task: 'test task',
    });
    expect(mockFetch).toHaveBeenCalledWith('/api/runs', expect.objectContaining({
      method: 'POST',
      body: expect.stringContaining('test task'),
    }));
  });

  it('getRun calls correct endpoint', async () => {
    const mockRun: Partial<RunState> = {
      run_id: 'run-abc',
      status: 'running',
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve(mockRun),
    });
    const result = await getRun('run-abc');
    expect(mockFetch).toHaveBeenCalledWith('/api/runs/run-abc', expect.objectContaining({
      headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
    }));
    expect(result.run_id).toBe('run-abc');
  });

  it('compareRuns sends POST with run_ids', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ metrics: { runs: [] } }),
    });
    await compareRuns(['id1', 'id2'], 'summary');
    expect(mockFetch).toHaveBeenCalledWith('/api/comparison/summary', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ run_ids: ['id1', 'id2'] }),
    }));
  });

  it('throws ApiError on non-ok response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: () => Promise.resolve('Not Found'),
    });
    await expect(listWorkspaces()).rejects.toThrow();
  });

  it('getKnowledgeSummary calls correct endpoint', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        total_claims: 5,
        total_facts: 10,
        total_links: 3,
        avg_uncertainty: 0.4,
        avg_strength: 0.6,
      }),
    });
    const result = await getKnowledgeSummary();
    expect(mockFetch).toHaveBeenCalledWith('/api/knowledge/summary', expect.anything());
    expect(result.total_claims).toBe(5);
  });
});
