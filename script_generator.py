import streamlit as st 
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun

# # Initialize Llama 3.3
# def get_llm(creativity):
#     return Ollama(
#         model="llama3.3:latest",  # Using Llama 3.3
#         temperature=creativity,
#         top_p=0.9,
#         top_k=40,
#         num_ctx=4096,  # Larger context window
#         repeat_penalty=1.1
#     )



# Change this line in the get_llm function
def get_llm(creativity):
    return Ollama(
        model="llama2:latest",  # Changed from llama3.3:latest to llama2:latest
        temperature=creativity,
        top_p=0.9,
        top_k=40,
        num_ctx=2048,  # Reduced context window for smaller model
        repeat_penalty=1.1
    )

def generate_script(prompt, video_length, creativity):
    # Initialize LLM
    llm = get_llm(creativity)
    
    # Enhanced template for generating 'Title'
    title_template = PromptTemplate(
        input_variables=['subject'], 
        template='''Create a YouTube video title that is:
        1. SEO-friendly
        2. Attention-grabbing
        3. Clear and concise
        
        Topic: {subject}
        
        Return only the title, no additional text.'''
    )

    # Enhanced template for generating 'Video Script'
    script_template = PromptTemplate(
        input_variables=['title', 'DuckDuckGo_Search', 'duration'], 
        template='''Create a professional YouTube video script for: {title}
        Target Duration: {duration} minutes
        
        Research Data: {DuckDuckGo_Search}
        
        Structure the script as follows:

        [HOOK]
        - Attention-grabbing opening (15 seconds)
        
        [INTRODUCTION]
        - Brief overview
        - What viewers will learn
        
        [MAIN CONTENT]
        - Divide into clear sections
        - Include timestamps
        - Add relevant examples
        - Include data points from research
        
        [CONCLUSION]
        - Summarize key points
        - Call to action
        
        Make it conversational, engaging, and well-paced for {duration} minutes.'''
    )

    # Create chains
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

    # Initialize search
    search = DuckDuckGoSearchRun()

    try:
        # Get search results
        with st.spinner('🔍 Researching your topic...'):
            search_result = search.run(prompt)

        # Generate title
        with st.spinner('✍️ Crafting the perfect title...'):
            title = title_chain.invoke({'subject': prompt})
            title = title['text'] if isinstance(title, dict) else title

        # Generate script
        with st.spinner('📝 Writing your script...'):
            script = script_chain.run(
                title=title,
                DuckDuckGo_Search=search_result,
                duration=video_length
            )

        return search_result, title, script

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None

def main():
    # Styling
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff;
        color:#ffffff;
        font-weight: bold;
        padding: 0.8em 1.2em;
        border-radius: 10px;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color:#FFFFFF;
        transform: scale(1.02);
    }
    .main-header {
        font-size: 2.5rem;
        color: #0099ff;
        text-align: center;
        margin-bottom: 1em;
    }
    .stSubheader {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main interface
    st.markdown("<h1 class='main-header'>🎥 YouTube Script Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='stSubheader'>Powered by Llama 3.3 - Advanced AI Script Writing</p>", unsafe_allow_html=True)

    # Input section
    with st.container():
        st.markdown("### 📝 Enter Your Video Details")
        prompt = st.text_input(
            'Video Topic',
            placeholder="E.g., 'The Future of Artificial Intelligence in Healthcare'",
            help="Be specific about what you want to cover in your video"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            video_length = st.number_input(
                'Video Length (minutes)',
                min_value=1,
                max_value=60,
                value=5,
                help="How long should your video be?"
            )
        with col2:
            creativity = st.slider(
                'Creativity Level',
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Higher values make the output more creative but less focused"
            )

    # Generate button
    if st.button("✨ Generate Script", type="primary"):
        if prompt:
            search_result, title, script = generate_script(prompt, video_length, creativity)
            
            if title and script:
                # Display results in a nice format
                st.markdown("---")
                st.markdown("### 🎬 Your Video Title")
                st.info(title)

                st.markdown("### 📄 Your Video Script")
                st.text_area("Script", script, height=400)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    script_text = f"Title: {title}\n\n{script}"
                    st.download_button(
                        label="📥 Download Script (TXT)",
                        data=script_text,
                        file_name="youtube_script.txt",
                        mime="text/plain"
                    )
                with col2:
                    # Create markdown version
                    markdown_text = f"# {title}\n\n{script}"
                    st.download_button(
                        label="📥 Download Script (MD)",
                        data=markdown_text,
                        file_name="youtube_script.md",
                        mime="text/markdown"
                    )

                # Show research data
                with st.expander("🔍 Research Data"):
                    st.markdown(search_result)
        else:
            st.warning("Please enter a topic for your video!")

    # Help section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Getting Started
        1. Enter your video topic in detail
        2. Set the desired video length
        3. Adjust the creativity level
        4. Click 'Generate Script'
        5. Download your script
        
        ### Tips for Best Results
        - Be specific with your topic
        - Start with 5-10 minute scripts
        - Use creativity level 0.7 for balanced output
        - Review and edit the generated script
        - Use the research data for additional insights
        
        ### About Llama 3.3
        This generator uses Llama 3.3, a powerful language model optimized for creative and technical writing.
        """)

if __name__ == "__main__":
    main()