#!/usr/bin/env node

/**
 * Spotify metadata builder for NeuroTunes (WP5).
 *
 * Usage:
 *   SPOTIFY_CLIENT_ID=... SPOTIFY_CLIENT_SECRET=... node tools/build_spotify.js
 */

const fs = require('fs');
const path = require('path');

const CLIENT_ID = process.env.SPOTIFY_CLIENT_ID;
const CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET;

if (!CLIENT_ID || !CLIENT_SECRET) {
  console.error('Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET environment variables.');
  process.exit(1);
}

const CLIP_QUERIES = {
  clip_01: 'Chim Chim Cheree lyrics',
  clip_02: 'Take Me Out To The Ball Game lyrics',
  clip_03: 'Jingle Bells lyrics',
  clip_04: 'Mary Had A Little Lamb lyrics',
  clip_11: 'Chim Chim Cheree instrumental',
  clip_12: 'Take Me Out To The Ball Game instrumental',
  clip_13: 'Jingle Bells instrumental',
  clip_14: 'Mary Had A Little Lamb instrumental',
  clip_21: 'Emperor Waltz',
  clip_22: 'Harry Potter Theme',
  clip_23: 'Star Wars Main Theme',
  clip_24: 'Eine kleine Nachtmusik',
};

const DATA_DIR = path.resolve(__dirname, '..', 'docs', 'data');
const SEARCH_OUT_PATH = path.join(DATA_DIR, 'spotify_search.json');
const RECS_OUT_PATH = path.join(DATA_DIR, 'playlist_recs.json');

async function getAccessToken() {
  const credentials = Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString('base64');
  const response = await fetch('https://accounts.spotify.com/api/token', {
    method: 'POST',
    headers: {
      Authorization: `Basic ${credentials}`,
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({ grant_type: 'client_credentials' }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to obtain token: ${response.status} ${text}`);
  }

  const data = await response.json();
  return data.access_token;
}

async function searchTracks(token, query, limit = 5) {
  const url = new URL('https://api.spotify.com/v1/search');
  url.searchParams.set('q', query);
  url.searchParams.set('type', 'track');
  url.searchParams.set('limit', String(limit));

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Search failed for "${query}": ${response.status} ${text}`);
  }

  const data = await response.json();
  return data.tracks.items || [];
}

function normaliseTrack(track) {
  return {
    id: track.id,
    name: track.name,
    artists: track.artists.map((a) => a.name).join(', '),
    url: track.external_urls?.spotify ?? null,
    image:
      track.album?.images?.length > 0
        ? track.album.images.sort((a, b) => b.width - a.width)[0].url
        : null,
    score: track.popularity ?? null,
  };
}

async function buildSearchCatalog(token) {
  const results = {};

  for (const [clipId, query] of Object.entries(CLIP_QUERIES)) {
    try {
      const items = await searchTracks(token, query, 5);
      results[clipId] = items.map(normaliseTrack);
    } catch (error) {
      console.warn(`Warning: ${error.message}`);
      results[clipId] = [];
    }
  }

  return results;
}

async function buildRecommendations(token, seeds) {
  if (seeds.length === 0) {
    return [];
  }

  const seedList = seeds.slice(0, 5);
  console.log('Seed track IDs:', seedList);

  const url = new URL('https://api.spotify.com/v1/recommendations');
  url.searchParams.set('limit', '12');
  url.searchParams.set('seed_tracks', seedList.join(','));
  url.searchParams.set('market', 'US');

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const text = await response.text();
    console.error(`Recommendations request failed: ${response.status}`);
    console.error('Response body:', text);
    throw new Error(`Recommendations request failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  return (data.tracks || []).map(normaliseTrack).slice(0, 12);
}

async function main() {
  try {
    const token = await getAccessToken();
    console.log('Access token obtained successfully.');

    const searchCatalog = await buildSearchCatalog(token);
    const seedIds = Object.values(searchCatalog)
      .flatMap((tracks) => (tracks.length > 0 ? [tracks[0].id] : []));

    let recommendations = [];
    try {
      recommendations = await buildRecommendations(token, seedIds);
    } catch (error) {
      console.warn('Recommendations failed, using empty list:', error.message);
    }

    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(SEARCH_OUT_PATH, JSON.stringify(searchCatalog, null, 2));
    fs.writeFileSync(RECS_OUT_PATH, JSON.stringify(recommendations, null, 2));

    console.log(`Wrote Spotify search results to ${SEARCH_OUT_PATH}`);
    console.log(`Wrote playlist recommendations to ${RECS_OUT_PATH}`);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}

main();
