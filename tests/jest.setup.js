// Jest setup file for Puppeteer tests
jest.setTimeout(30000);

// Global test setup
beforeAll(() => {
    console.log('Starting E2E test suite...');
});

afterAll(() => {
    console.log('E2E test suite completed');
});