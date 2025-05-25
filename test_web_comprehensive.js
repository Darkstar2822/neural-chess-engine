/**
 * Comprehensive Web Application Testing with Puppeteer
 * Tests all functionality including evolved models, API endpoints, and UI interactions
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

class WebAppTester {
    constructor() {
        this.browser = null;
        this.page = null;
        this.baseUrl = 'http://127.0.0.1:5000';
        this.screenshotDir = 'test_screenshots';
        this.results = {
            passed: 0,
            failed: 0,
            issues: []
        };
    }

    async setup() {
        // Create screenshot directory
        if (!fs.existsSync(this.screenshotDir)) {
            fs.mkdirSync(this.screenshotDir, { recursive: true });
        }

        // Launch browser
        this.browser = await puppeteer.launch({
            headless: false, // Show browser for debugging
            defaultViewport: { width: 1200, height: 800 },
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        this.page = await this.browser.newPage();
        
        // Set up console logging
        this.page.on('console', msg => {
            console.log(`🌐 BROWSER: ${msg.text()}`);
        });

        // Set up error handling
        this.page.on('error', err => {
            console.error(`❌ PAGE ERROR: ${err.message}`);
            this.addIssue('Page Error', err.message);
        });

        this.page.on('pageerror', err => {
            console.error(`❌ PAGE ERROR: ${err.message}`);
            this.addIssue('Page Error', err.message);
        });
    }

    async takeScreenshot(name, description = '') {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `${timestamp}_${name}.png`;
        const filepath = path.join(this.screenshotDir, filename);
        
        await this.page.screenshot({ 
            path: filepath, 
            fullPage: true 
        });
        
        console.log(`📸 Screenshot saved: ${filename} - ${description}`);
        return filepath;
    }

    addIssue(type, description) {
        this.results.issues.push({
            type,
            description,
            timestamp: new Date().toISOString()
        });
    }

    async testApiEndpoint(endpoint, expectedStatus = 200) {
        try {
            const response = await this.page.goto(`${this.baseUrl}${endpoint}`);
            const status = response.status();
            
            if (status === expectedStatus) {
                console.log(`✅ API ${endpoint}: ${status}`);
                this.results.passed++;
                return true;
            } else {
                console.log(`❌ API ${endpoint}: Expected ${expectedStatus}, got ${status}`);
                this.addIssue('API Error', `${endpoint} returned ${status}, expected ${expectedStatus}`);
                this.results.failed++;
                return false;
            }
        } catch (error) {
            console.log(`❌ API ${endpoint}: ${error.message}`);
            this.addIssue('API Error', `${endpoint} failed: ${error.message}`);
            this.results.failed++;
            return false;
        }
    }

    async testApiEndpoints() {
        console.log('\n🔌 Testing API Endpoints...');
        
        const endpoints = [
            '/api/health',
            '/api/models',
            '/api/current_model',
            '/api/best_model',
            '/api/game_status',
            '/api/legal_moves'
        ];

        for (const endpoint of endpoints) {
            await this.testApiEndpoint(endpoint);
            await new Promise(resolve => setTimeout(resolve, 500)); // Rate limiting
        }
    }

    async testMainInterface() {
        console.log('\n🏠 Testing Main Interface...');
        
        try {
            await this.page.goto(this.baseUrl);
            await this.page.waitForSelector('body', { timeout: 10000 });
            
            await this.takeScreenshot('01_main_interface', 'Main chess interface loaded');
            
            // Check for essential elements
            const elements = [
                '#chess-board',
                '.game-controls',
                '.model-info'
            ];

            for (const selector of elements) {
                const element = await this.page.$(selector);
                if (element) {
                    console.log(`✅ Found element: ${selector}`);
                    this.results.passed++;
                } else {
                    console.log(`❌ Missing element: ${selector}`);
                    this.addIssue('UI Element Missing', `Required element ${selector} not found`);
                    this.results.failed++;
                }
            }

            // Check page title
            const title = await this.page.title();
            if (title.includes('Chess')) {
                console.log(`✅ Page title: ${title}`);
                this.results.passed++;
            } else {
                console.log(`❌ Unexpected page title: ${title}`);
                this.addIssue('UI Issue', `Page title should contain 'Chess', got: ${title}`);
                this.results.failed++;
            }

        } catch (error) {
            console.log(`❌ Main interface test failed: ${error.message}`);
            this.addIssue('UI Error', `Main interface failed to load: ${error.message}`);
            this.results.failed++;
        }
    }

    async testChessBoard() {
        console.log('\n♟️ Testing Chess Board...');
        
        try {
            // Check if chess board is visible
            await this.page.waitForSelector('#chess-board', { timeout: 5000 });
            
            // Take screenshot of chess board
            await this.takeScreenshot('02_chess_board', 'Chess board display');
            
            // Check for chess pieces
            const pieces = await this.page.$$('.chess-piece, .piece, [data-piece]');
            if (pieces.length > 0) {
                console.log(`✅ Found ${pieces.length} chess pieces on board`);
                this.results.passed++;
            } else {
                console.log(`❌ No chess pieces found on board`);
                this.addIssue('Chess Board', 'No chess pieces visible on the board');
                this.results.failed++;
            }

            // Check board squares
            const squares = await this.page.$$('.square, [data-square], .chess-square');
            if (squares.length >= 64) {
                console.log(`✅ Found ${squares.length} board squares`);
                this.results.passed++;
            } else {
                console.log(`❌ Expected 64+ squares, found ${squares.length}`);
                this.addIssue('Chess Board', `Expected 64+ squares, found ${squares.length}`);
                this.results.failed++;
            }

        } catch (error) {
            console.log(`❌ Chess board test failed: ${error.message}`);
            this.addIssue('Chess Board Error', error.message);
            this.results.failed++;
        }
    }

    async testGameControls() {
        console.log('\n🎮 Testing Game Controls...');
        
        try {
            // Look for new game button
            const newGameButton = await this.page.$('#new-game, .new-game, button[onclick*="newGame"]') || 
                                    await this.page.$xpath('//button[contains(text(), "New Game") or contains(text(), "Play as")]');
            if (newGameButton) {
                console.log('✅ Found new game button');
                this.results.passed++;
                
                // Test clicking new game
                await newGameButton.click();
                await new Promise(resolve => setTimeout(resolve, 1000));
                await this.takeScreenshot('03_new_game_clicked', 'After clicking new game');
                
            } else {
                console.log('❌ New game button not found');
                this.addIssue('Game Controls', 'New game button not found');
                this.results.failed++;
            }

            // Look for color selection
            const colorButtons = await this.page.$$('button[data-color], .color-select button, input[type="radio"][name*="color"]');
            if (colorButtons.length >= 2) {
                console.log(`✅ Found ${colorButtons.length} color selection controls`);
                this.results.passed++;
            } else {
                console.log(`❌ Expected 2+ color controls, found ${colorButtons.length}`);
                this.addIssue('Game Controls', 'Color selection controls not found');
                this.results.failed++;
            }

        } catch (error) {
            console.log(`❌ Game controls test failed: ${error.message}`);
            this.addIssue('Game Controls Error', error.message);
            this.results.failed++;
        }
    }

    async testModelInfo() {
        console.log('\n🧠 Testing Model Information...');
        
        try {
            // Look for model information display
            const modelInfo = await this.page.$('.model-info, #model-info, .current-model');
            if (modelInfo) {
                console.log('✅ Found model info section');
                this.results.passed++;
                
                const modelText = await this.page.evaluate(el => el.textContent, modelInfo);
                console.log(`📊 Model info: ${modelText.substring(0, 100)}...`);
                
            } else {
                console.log('❌ Model info section not found');
                this.addIssue('Model Info', 'Model information section not found');
                this.results.failed++;
            }

            // Test model API endpoint
            const response = await this.page.evaluate(async () => {
                try {
                    const res = await fetch('/api/current_model');
                    return await res.json();
                } catch (error) {
                    return { error: error.message };
                }
            });

            if (response.error) {
                console.log(`❌ Model API error: ${response.error}`);
                this.addIssue('Model API', response.error);
                this.results.failed++;
            } else {
                console.log(`✅ Model API response: ${response.architecture || 'Unknown'}`);
                this.results.passed++;
            }

        } catch (error) {
            console.log(`❌ Model info test failed: ${error.message}`);
            this.addIssue('Model Info Error', error.message);
            this.results.failed++;
        }
    }

    async testMoveExecution() {
        console.log('\n🎯 Testing Move Execution...');
        
        try {
            // Start a new game first
            await this.page.evaluate(() => {
                if (window.newGame) {
                    window.newGame('white');
                } else if (window.startNewGame) {
                    window.startNewGame('white');
                }
            });

            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Try to make a move (e2 to e4)
            const moveResult = await this.page.evaluate(async () => {
                try {
                    if (window.makeMove) {
                        return await window.makeMove('e2e4');
                    } else if (window.makeUserMove) {
                        return await window.makeUserMove('e2e4');
                    } else {
                        // Try API directly
                        const response = await fetch('/api/make_move', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ move: 'e2e4' })
                        });
                        return await response.json();
                    }
                } catch (error) {
                    return { error: error.message };
                }
            });

            if (moveResult && !moveResult.error) {
                console.log('✅ Move execution successful');
                this.results.passed++;
                await this.takeScreenshot('04_move_executed', 'After executing move e2e4');
            } else {
                console.log(`❌ Move execution failed: ${moveResult?.error || 'Unknown error'}`);
                this.addIssue('Move Execution', moveResult?.error || 'Move execution failed');
                this.results.failed++;
            }

        } catch (error) {
            console.log(`❌ Move execution test failed: ${error.message}`);
            this.addIssue('Move Execution Error', error.message);
            this.results.failed++;
        }
    }

    async testResponsiveness() {
        console.log('\n📱 Testing Responsiveness...');
        
        const viewports = [
            { width: 1920, height: 1080, name: 'Desktop' },
            { width: 768, height: 1024, name: 'Tablet' },
            { width: 375, height: 667, name: 'Mobile' }
        ];

        for (const viewport of viewports) {
            try {
                await this.page.setViewport(viewport);
                await new Promise(resolve => setTimeout(resolve, 500));
                
                await this.takeScreenshot(`05_responsive_${viewport.name.toLowerCase()}`, 
                    `${viewport.name} view ${viewport.width}x${viewport.height}`);
                
                // Check if essential elements are still visible
                const boardVisible = await this.page.$('#chess-board');
                if (boardVisible) {
                    console.log(`✅ ${viewport.name}: Chess board visible`);
                    this.results.passed++;
                } else {
                    console.log(`❌ ${viewport.name}: Chess board not visible`);
                    this.addIssue('Responsiveness', `Chess board not visible in ${viewport.name} view`);
                    this.results.failed++;
                }

            } catch (error) {
                console.log(`❌ ${viewport.name} test failed: ${error.message}`);
                this.addIssue('Responsiveness Error', `${viewport.name}: ${error.message}`);
                this.results.failed++;
            }
        }

        // Reset to desktop view
        await this.page.setViewport({ width: 1200, height: 800 });
    }

    async testPerformance() {
        console.log('\n⚡ Testing Performance...');
        
        try {
            // Measure page load time
            const startTime = Date.now();
            await this.page.goto(this.baseUrl, { waitUntil: 'networkidle0' });
            const loadTime = Date.now() - startTime;
            
            console.log(`📊 Page load time: ${loadTime}ms`);
            
            if (loadTime < 5000) {
                console.log('✅ Page load time acceptable');
                this.results.passed++;
            } else {
                console.log('❌ Page load time too slow');
                this.addIssue('Performance', `Page load time ${loadTime}ms exceeds 5000ms`);
                this.results.failed++;
            }

            // Check for JavaScript errors
            const jsErrors = [];
            this.page.on('pageerror', error => {
                jsErrors.push(error.message);
            });

            if (jsErrors.length === 0) {
                console.log('✅ No JavaScript errors detected');
                this.results.passed++;
            } else {
                console.log(`❌ Found ${jsErrors.length} JavaScript errors`);
                jsErrors.forEach(error => {
                    this.addIssue('JavaScript Error', error);
                });
                this.results.failed++;
            }

        } catch (error) {
            console.log(`❌ Performance test failed: ${error.message}`);
            this.addIssue('Performance Error', error.message);
            this.results.failed++;
        }
    }

    async runAllTests() {
        console.log('🚀 Starting Comprehensive Web App Testing\n');
        
        try {
            await this.setup();
            
            // Test API endpoints first
            await this.testApiEndpoints();
            
            // Test main interface
            await this.testMainInterface();
            
            // Test chess board
            await this.testChessBoard();
            
            // Test game controls
            await this.testGameControls();
            
            // Test model information
            await this.testModelInfo();
            
            // Test move execution
            await this.testMoveExecution();
            
            // Test responsiveness
            await this.testResponsiveness();
            
            // Test performance
            await this.testPerformance();
            
            // Final screenshot
            await this.takeScreenshot('06_final_state', 'Final application state');
            
        } catch (error) {
            console.error(`❌ Test suite failed: ${error.message}`);
            this.addIssue('Test Suite Error', error.message);
        } finally {
            await this.cleanup();
        }
    }

    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
        this.generateReport();
    }

    generateReport() {
        console.log('\n📋 TEST RESULTS SUMMARY');
        console.log('=' * 50);
        console.log(`✅ Passed: ${this.results.passed}`);
        console.log(`❌ Failed: ${this.results.failed}`);
        console.log(`📊 Total: ${this.results.passed + this.results.failed}`);
        
        if (this.results.issues.length > 0) {
            console.log('\n🐛 ISSUES FOUND:');
            this.results.issues.forEach((issue, index) => {
                console.log(`${index + 1}. [${issue.type}] ${issue.description}`);
            });
        } else {
            console.log('\n🎉 No issues found!');
        }

        // Save report to file
        const reportPath = path.join(this.screenshotDir, 'test_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));
        console.log(`\n📄 Detailed report saved: ${reportPath}`);
        console.log(`📸 Screenshots saved in: ${this.screenshotDir}/`);
    }
}

// Run the tests
const tester = new WebAppTester();
tester.runAllTests().catch(console.error);