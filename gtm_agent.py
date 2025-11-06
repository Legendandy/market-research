import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler
)
from typing import AsyncIterator
from app.core import (
    run_smartcrawler,
    run_searchscraper,
    llm,  # Import LLM directly instead of agents
    extract_company_name,
    normalize_url
)

load_dotenv("api.env")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SmartGTMAgent(AbstractAgent):
    """
    Smart GTM Agent - AI-powered Go-To-Market Strategy Assistant
    
    Compliant with Sentient Agent Framework for lightning-fast market 
    intelligence & GTM execution.
    """
    
    def __init__(self, name: str = "Smart GTM Agent"):
        super().__init__(name)
        
        # Validate API keys
        self.nebius_key = os.getenv("NEBIUS_API_KEY")
        self.smartcrawler_key = os.getenv("SMARTCRAWLER_API_KEY")
        
        if not self.nebius_key:
            raise ValueError("NEBIUS_API_KEY is not set in environment")
        if not self.smartcrawler_key:
            raise ValueError("SMARTCRAWLER_API_KEY is not set in environment")
        
        # Create thread pool executor for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info(f"✅ {name} initialized successfully")
    
    async def _run_with_keepalive(self, func, response_handler, status_key, *args):
        """Run a blocking function with periodic keepalive messages"""
        loop = asyncio.get_event_loop()
        
        # Create the main task
        task = loop.run_in_executor(self.executor, func, *args)
        
        # Create keepalive task
        async def send_keepalive():
            counter = 0
            while not task.done():
                await asyncio.sleep(15)  # Send every 15 seconds
                if not task.done():
                    counter += 1
                    await response_handler.emit_text_block(
                        f"{status_key}_KEEPALIVE_{counter}",
                        f"⏳ Still processing {status_key.lower()}... ({counter * 15}s elapsed)\n"
                    )
        
        # Run both tasks concurrently
        keepalive_task = asyncio.create_task(send_keepalive())
        
        try:
            result = await task
            keepalive_task.cancel()
            return result
        except Exception as e:
            keepalive_task.cancel()
            raise e
    
    async def assist(
        self,
        session: Session,
        query: Query,
        response_handler: ResponseHandler
    ):
        """
        Main entry point for the GTM agent.
        
        Expected query format:
        {
            "prompt": "company_url",
            "feature": "research|go-to-market|channel"
        }
        """
        try:
            # Parse query - support formats:
            # 1. "https://example.com" (defaults to research)
            # 2. "https://example.com | research"
            # 3. "https://example.com | go-to-market"
            # 4. "https://example.com | channel"
            
            prompt_parts = query.prompt.strip().split('|')
            company_url = prompt_parts[0].strip()
            
            # Normalize URL
            company_url = normalize_url(company_url)
            
            # Extract feature from prompt or query metadata
            if len(prompt_parts) > 1:
                feature = prompt_parts[1].strip().lower()
            else:
                feature = getattr(query, 'feature', 'research').lower()
            
            # Validate inputs
            if not company_url or not company_url.startswith('http'):
                await response_handler.emit_error(
                    error_code=400,
                    error_data={"message": "Please provide a valid company URL starting with http/https"}
                )
                await response_handler.complete()
                return
            
            if feature not in ['research', 'go-to-market', 'channel']:
                await response_handler.emit_error(
                    error_code=400,
                    error_data={"message": f"Invalid feature '{feature}'. Must be one of: research, go-to-market, channel"}
                )
                await response_handler.complete()
                return
            
            # Extract company name
            company_name = extract_company_name(company_url)
            
            # IMMEDIATE RESPONSE - prevent timeout
            await response_handler.emit_text_block(
                "ANALYSIS_START",
                f"🚀 Starting {feature.upper()} analysis for {company_name}...\n\n"
                "⏳ This process may take 2-5 minutes. Please wait...\n"
            )
            
            # Run data collection with progress updates
            await response_handler.emit_text_block(
                "DATA_COLLECTION",
                "🕷️ Step 1/3: Running SmartCrawler for company data extraction...\n"
                "⏱️ Estimated time: 2-3 minutes\n"
            )
            
            try:
                # Run SmartCrawler with keepalive
                scrawler_result = await self._run_with_keepalive(
                    run_smartcrawler,
                    response_handler,
                    "SMARTCRAWLER",
                    company_url
                )
                
                await response_handler.emit_json(
                    "SMARTCRAWLER_COMPLETE",
                    {"status": "success", "data_length": len(scrawler_result)}
                )
                
                await response_handler.emit_text_block(
                    "COMPETITOR_SEARCH",
                    "🔍 Step 2/3: Running SearchScraper for competitor analysis...\n"
                    "⏱️ Estimated time: 1-2 minutes\n"
                )
                
                # Run SearchScraper with keepalive
                search_result = await self._run_with_keepalive(
                    run_searchscraper,
                    response_handler,
                    "SEARCHSCRAPER",
                    scrawler_result  # Pass the company data, not URL
                )
                
                await response_handler.emit_json(
                    "SEARCHSCRAPER_COMPLETE",
                    {"status": "success", "data_length": len(search_result)}
                )
                
                # Combine results
                combined_data = f"## 🕷️ Crawler Data:\n{scrawler_result}\n\n## 🔍 Scraper Data:\n{search_result}"
                
                # Process based on selected feature
                await response_handler.emit_text_block(
                    "AGENT_PROCESSING",
                    f"🤖 Step 3/3: Running {feature.upper()} analysis...\n"
                )
                
                # Create text stream for final response
                final_response_stream = response_handler.create_text_stream(
                    "FINAL_RESPONSE"
                )
                
                # Process with appropriate analysis
                if feature == "research":
                    async for chunk in self._run_research_analysis(combined_data, response_handler):
                        await final_response_stream.emit_chunk(chunk)
                        
                elif feature == "go-to-market":
                    async for chunk in self._run_gtm_analysis(combined_data, response_handler):
                        await final_response_stream.emit_chunk(chunk)
                        
                elif feature == "channel":
                    async for chunk in self._run_channel_analysis(combined_data, response_handler):
                        await final_response_stream.emit_chunk(chunk)
                
                await final_response_stream.complete()
                
                # Completion metadata
                await response_handler.emit_json(
                    "ANALYSIS_METADATA",
                    {
                        "company_url": company_url,
                        "company_name": company_name,
                        "feature": feature,
                        "status": "completed"
                    }
                )
                
            except Exception as e:
                logger.error(f"Error during processing: {e}", exc_info=True)
                raise
            
            await response_handler.complete()
            logger.info(f"✅ Analysis completed for {company_url}")
            
        except Exception as e:
            logger.error(f"Error in assist method: {e}", exc_info=True)
            await response_handler.emit_error(
                error_code=500,
                error_data={"message": f"An error occurred: {str(e)}"}
            )
            await response_handler.complete()
    
    async def _run_research_analysis(self, context: str, response_handler: ResponseHandler) -> AsyncIterator[str]:
        """Run research analysis and stream results"""
        try:
            prompt = f"""You are a professional Company Research & Market Intelligence Assistant.

Analyze the provided company and competitor data and create a comprehensive research report.

Structure your report with these sections:

1. **Company Overview** – history, mission, vision, key offerings
2. **Founders & Leadership** – key people and their background
3. **Funding & Financials** – funding rounds, investors, financial health
4. **Industry & Market Size** – sector, growth rate, TAM/SAM/SOM if available
5. **Competitors** – top direct & indirect competitors with brief comparison
6. **Market Insights & Trends** – opportunities, risks, and emerging trends
7. **Assumptions & Gaps** – list any missing or uncertain information

Guidelines:
- Be concise, factual, and business-ready
- Use bullet points where appropriate
- Include units (USD, %, year) for numbers
- State uncertainty explicitly when data is missing or unclear
- Focus on actionable insights for strategic decision-making

DATA TO ANALYZE:
{context}

Provide your comprehensive research report below:
"""
            
            # Run LLM in executor
            loop = asyncio.get_event_loop()
            
            def _invoke():
                response = llm.invoke(prompt)
                return response.content
            
            result = await loop.run_in_executor(self.executor, _invoke)
            
            # Stream the result in chunks
            chunk_size = 100
            for i in range(0, len(result), chunk_size):
                yield result[i:i + chunk_size]
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
        except Exception as e:
            logger.error(f"Research analysis error: {e}", exc_info=True)
            yield f"\n\n❌ Error in research analysis: {str(e)}"
    
    async def _run_gtm_analysis(self, context: str, response_handler: ResponseHandler) -> AsyncIterator[str]:
        """Run GTM analysis and stream results"""
        try:
            prompt = f"""You are a professional Go-To-Market (GTM) Strategist.

Based on the provided company and competitor data, create a comprehensive GTM strategy.

Your GTM Playbook must include these sections:

1. **Executive Summary** – 2-3 sentences summarizing the GTM approach
2. **Target Market Analysis** – market segments, size, opportunities, positioning
3. **Ideal Customer Profile (ICP)** – demographics, firmographics, pain points, buying behavior
4. **Core Messaging & Value Proposition** – key narratives, positioning statements, differentiators
5. **Pricing Strategy** – pricing model, competitive positioning, justification
6. **Distribution & Sales Strategy** – direct/indirect channels, sales motion, partner ecosystem
7. **Growth Channels & Tactics** – short-term & long-term acquisition channels (SEO, paid ads, partnerships, etc.)
8. **Metrics & KPIs** – 5-8 measurable success indicators to track
9. **Assumptions & Risks** – key assumptions made and potential risks to mitigate

Output Requirements:
- Professional, actionable, presentation-ready format
- Clear structure with headers and bullet points
- Specific, concrete recommendations (not generic advice)
- Base all recommendations on the provided data
- Be realistic about what can be achieved given the company's stage and resources

DATA TO ANALYZE:
{context}

Provide your comprehensive GTM strategy below:
"""
            
            # Run LLM in executor
            loop = asyncio.get_event_loop()
            
            def _invoke():
                response = llm.invoke(prompt)
                return response.content
            
            result = await loop.run_in_executor(self.executor, _invoke)
            
            # Stream the result in chunks
            chunk_size = 100
            for i in range(0, len(result), chunk_size):
                yield result[i:i + chunk_size]
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"GTM analysis error: {e}", exc_info=True)
            yield f"\n\n❌ Error in GTM analysis: {str(e)}"
    
    async def _run_channel_analysis(self, context: str, response_handler: ResponseHandler) -> AsyncIterator[str]:
        """Run channel analysis and stream results"""
        try:
            prompt = f"""You are a Distribution & Channel Strategy Expert.

Based on the provided company and competitor data, recommend optimal distribution channels.

Your Channel Strategy must cover:

1. **Channel Strategy Overview** – 2-3 sentences on overall approach
2. **Primary Channels** – direct sales, online platforms, retail, partnerships
3. **Digital Channels** – SEO, paid ads, marketplaces, app stores, social media, content marketing
4. **Partnerships & Alliances** – distributors, affiliates, VARs, strategic integrations
5. **Emerging/Innovative Channels** – communities, niche platforms, unconventional approaches
6. **Channel Prioritization Matrix** – which channels to pursue first and why
7. **Channel Economics** – estimated CAC, LTV, and ROI by channel (if data allows)
8. **Implementation Roadmap** – 90-day, 180-day, and 365-day milestones
9. **Success Metrics** – KPIs for measuring channel performance
10. **Risks & Dependencies** – channel-specific risks and mitigation strategies

Output Requirements:
- Well-structured, business-oriented, actionable guidance
- Clear format with headers and bullet points
- Specific channel recommendations with justification based on:
  * Company stage and resources
  * Product/service fit
  * Target customer behavior
  * Competitive landscape
- Include both immediate quick wins and long-term strategic channels

DATA TO ANALYZE:
{context}

Provide your comprehensive channel strategy below:
"""
            
            # Run LLM in executor
            loop = asyncio.get_event_loop()
            
            def _invoke():
                response = llm.invoke(prompt)
                return response.content
            
            result = await loop.run_in_executor(self.executor, _invoke)
            
            # Stream the result in chunks
            chunk_size = 100
            for i in range(0, len(result), chunk_size):
                yield result[i:i + chunk_size]
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Channel analysis error: {e}", exc_info=True)
            yield f"\n\n❌ Error in channel analysis: {str(e)}"
    
    def __del__(self):
        """Cleanup executor on agent destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


if __name__ == "__main__":
    # Create an instance of SmartGTMAgent
    agent = SmartGTMAgent(name="Smart GTM Agent")
    
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    
    # Run the server (default port 8080)
    logger.info("🚀 Starting Smart GTM Agent server...")
    server.run()