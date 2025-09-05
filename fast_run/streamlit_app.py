#!/usr/bin/env python3
"""
Streamlit frontend for 10-K Knowledge Base.

Provides an interactive web interface for searching and analyzing 10-K filings.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
from typing import List, Dict, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="10-K Knowledge Base",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

class APIClient:
    """Client for interacting with the 10-K Knowledge Base API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def search_filings(self, query: str, tickers: List[str] = None, 
                      years: List[int] = None, sections: List[str] = None, 
                      limit: int = 10) -> List[Dict]:
        """Search filings via API."""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "tickers": tickers,
                    "years": years,
                    "sections": sections,
                    "limit": limit
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return []
    
    def get_company_overview(self, ticker: str) -> Dict:
        """Get company overview via API."""
        try:
            response = requests.get(f"{self.base_url}/company/{ticker}", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {}
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics via API."""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return {}
    
    def get_available_tickers(self) -> List[str]:
        """Get available tickers via API."""
        try:
            response = requests.get(f"{self.base_url}/tickers", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return []
    
    def get_available_sections(self) -> List[str]:
        """Get available sections via API."""
        try:
            response = requests.get(f"{self.base_url}/sections", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return []

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()

# Cache API calls for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_stats():
    return api_client.get_stats()

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_tickers():
    return api_client.get_available_tickers()

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def get_cached_sections():
    return api_client.get_available_sections()

# Initialize API client
api_client = get_api_client()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .company-tag {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .section-tag {
        background-color: #ff7f0e;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 10-K Knowledge Base</h1>', unsafe_allow_html=True)
    st.markdown("### Search and analyze SEC 10-K filings with AI-powered insights")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Search", "📈 Analytics", "🏢 Company Overview", "ℹ️ About"]
    )
    
    if page == "🔍 Search":
        search_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "🏢 Company Overview":
        company_overview_page()
    else:
        about_page()

def search_page():
    """Main search interface."""
    
    st.header("Search 10-K Filings")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your search query",
                placeholder="e.g., artificial intelligence, climate risk, supply chain...",
                help="Search across all 10-K filing content"
            )
        
        with col2:
            limit = st.selectbox("Results", [10, 25, 50, 100], index=0)
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tickers = get_cached_tickers()
                selected_tickers = st.multiselect(
                    "Companies",
                    options=tickers,
                    help="Filter by specific companies"
                )
            
            with col2:
                current_year = datetime.now().year
                years_range = list(range(2020, current_year + 1))
                selected_years = st.multiselect(
                    "Years",
                    options=years_range,
                    help="Filter by fiscal years"
                )
            
            with col3:
                sections = get_cached_sections()
                # Clean up section names for display
                section_display = {s: s.replace('section_', '').replace('_', ' ').title() for s in sections}
                selected_sections = st.multiselect(
                    "Sections",
                    options=list(section_display.keys()),
                    format_func=lambda x: section_display.get(x, x),
                    help="Filter by 10-K sections"
                )
        
        submitted = st.form_submit_button("🔍 Search", type="primary")
    
    # Perform search
    if submitted and query:
        with st.spinner("Searching filings..."):
            results = api_client.search_filings(
                query=query,
                tickers=selected_tickers if selected_tickers else None,
                years=selected_years if selected_years else None,
                sections=selected_sections if selected_sections else None,
                limit=limit
            )
        
        if results:
            st.success(f"Found {len(results)} results")
            display_search_results(results, query)
        else:
            st.warning("No results found. Try adjusting your search terms or filters.")
    
    elif submitted and not query:
        st.error("Please enter a search query.")

def display_search_results(results: List[Dict], query: str):
    """Display search results in a nice format."""
    
    for i, result in enumerate(results, 1):
        with st.container():
            # Create a card-like layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Company and section tags
                company_tag = f'<span class="company-tag">{result["ticker"]}</span>' if result["ticker"] else ""
                section_name = result["section_name"].replace('section_', '').replace('_', ' ').title()
                section_tag = f'<span class="section-tag">{section_name}</span>'
                
                st.markdown(f"""
                <div class="result-card">
                    <div>
                        {company_tag}
                        {section_tag}
                        <strong>{result["company_name"] or result["ticker"] or "Unknown Company"}</strong>
                        <br>
                        <small>Fiscal Year: {result["fiscal_year"]} | Filing Date: {result["filing_date"]}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Content preview with highlighted query
                content_preview = result["content_preview"]
                if query.lower() in content_preview.lower():
                    # Simple highlighting (could be improved)
                    highlighted = content_preview.replace(
                        query, f"**{query}**"
                    )
                    st.markdown(highlighted)
                else:
                    st.text(content_preview)
            
            with col2:
                st.metric("Content Length", f"{result['content_length']:,} chars")
                
                # Add expand button for full content
                if st.button(f"📄 View Details", key=f"detail_{i}"):
                    st.session_state[f"show_detail_{i}"] = not st.session_state.get(f"show_detail_{i}", False)
            
            # Show detailed view if expanded
            if st.session_state.get(f"show_detail_{i}", False):
                st.markdown("**Full Content:**")
                st.text_area(
                    "Content",
                    value=result.get("content_preview", "Content not available"),
                    height=200,
                    key=f"content_{i}",
                    disabled=True
                )
            
            st.divider()

def analytics_page():
    """Analytics and statistics page."""
    
    st.header("Knowledge Base Analytics")
    
    # Get stats
    with st.spinner("Loading analytics..."):
        stats = get_cached_stats()
    
    if not stats:
        st.error("Unable to load analytics data.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Filings", f"{stats['total_filings']:,}")
    
    with col2:
        st.metric("Companies", f"{stats['unique_companies']:,}")
    
    with col3:
        if stats['total_text_length']:
            st.metric("Total Text", f"{stats['total_text_length'] / 1e6:.1f}M chars")
        else:
            st.metric("Total Text", "N/A")
    
    with col4:
        st.metric("Sections", f"{stats['total_sections']:,}")
    
    # Date range
    if stats['date_range'] and stats['date_range'][0] and stats['date_range'][1]:
        st.info(f"📅 Filing Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Companies by Filing Count")
        if stats['top_companies']:
            companies_df = pd.DataFrame(
                list(stats['top_companies'].items()),
                columns=['Company', 'Filings']
            )
            fig = px.bar(companies_df, x='Filings', y='Company', orientation='h')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Most Common Sections")
        if stats['top_sections']:
            sections_df = pd.DataFrame(
                list(stats['top_sections'].items()),
                columns=['Section', 'Count']
            )
            # Clean up section names
            sections_df['Section'] = sections_df['Section'].str.replace('section_', '').str.replace('_', ' ').str.title()
            fig = px.pie(sections_df, values='Count', names='Section')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def company_overview_page():
    """Company-specific overview page."""
    
    st.header("Company Overview")
    
    # Company selector
    tickers = get_cached_tickers()
    selected_ticker = st.selectbox(
        "Select a company",
        options=tickers,
        help="Choose a company to view detailed filing information"
    )
    
    if selected_ticker:
        with st.spinner(f"Loading data for {selected_ticker}..."):
            overview = api_client.get_company_overview(selected_ticker)
        
        if overview:
            # Company header
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{overview['company_name'] or selected_ticker}")
                st.write(f"**Ticker:** {overview['ticker']}")
                
                if overview['filing_date_range']:
                    st.write(f"**Filing Period:** {overview['filing_date_range'][0]} to {overview['filing_date_range'][1]}")
            
            with col2:
                st.metric("Total Filings", overview['total_filings'])
                if overview['avg_filing_length']:
                    st.metric("Avg Filing Length", f"{overview['avg_filing_length']:,.0f} chars")
            
            # Yearly filings chart
            if overview['yearly_filings']:
                st.subheader("Filings by Year")
                yearly_df = pd.DataFrame(
                    list(overview['yearly_filings'].items()),
                    columns=['Year', 'Count']
                )
                yearly_df = yearly_df.sort_values('Year')
                
                fig = px.line(yearly_df, x='Year', y='Count', markers=True)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Section availability
            if overview['section_availability']:
                st.subheader("Available Sections")
                sections_df = pd.DataFrame(
                    list(overview['section_availability'].items()),
                    columns=['Section', 'Count']
                )
                sections_df['Section'] = sections_df['Section'].str.replace('section_', '').str.replace('_', ' ').str.title()
                sections_df = sections_df.sort_values('Count', ascending=False)
                
                st.dataframe(sections_df, use_container_width=True)
        else:
            st.error(f"No data found for {selected_ticker}")

def about_page():
    """About page with information about the knowledge base."""
    
    st.header("About the 10-K Knowledge Base")
    
    st.markdown("""
    ### Overview
    This knowledge base contains structured data from SEC 10-K filings, enabling fast and powerful search capabilities 
    across corporate disclosures.
    
    ### Features
    - **Full-text search** across all 10-K content
    - **Section-specific filtering** (Business, Risk Factors, MD&A, etc.)
    - **Company and year filtering** for targeted analysis
    - **Fast query performance** powered by DuckDB
    - **Interactive analytics** and visualizations
    
    ### Data Sources
    - SEC EDGAR database
    - 10-K annual reports (non-amended filings)
    - Structured section extraction
    - Company metadata and filing dates
    
    ### Technology Stack
    - **Backend:** FastAPI + DuckDB + Parquet
    - **Frontend:** Streamlit
    - **Data Processing:** BeautifulSoup + pandas
    - **Storage:** Optimized columnar format for analytics
    
    ### Usage Tips
    - Use specific terms for better results (e.g., "artificial intelligence" vs "AI")
    - Combine filters to narrow down results
    - Check the Analytics page for dataset overview
    - Use Company Overview for deep-dives into specific companies
    """)
    
    # API status check
    st.subheader("System Status")
    try:
        stats = api_client.get_stats()
        if stats:
            st.success("✅ API connection successful")
            st.info(f"📊 Knowledge base contains {stats['total_filings']} filings from {stats['unique_companies']} companies")
        else:
            st.error("❌ Unable to connect to API")
    except Exception as e:
        st.error(f"❌ API Error: {e}")

if __name__ == "__main__":
    main()