import google.genai as genai
import os
from dotenv import load_dotenv
import ffmpeg

print(help(ffmpeg))
load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

myfile = client.files.upload(file="path/to/sample.mp4")

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents=[
        myfile, 
        """
        CONTEXT
            You are analyzing video footage from smart glasses worn by someone building an AI second brain. Your job is to understand what they're working on or learning and proactively enhance their knowledge.
            YOUR ROLE
            Identify the user's work/learning activities and provide intelligent research and insights to deepen their understanding and accelerate their progress.
        
        CORE TASKS
        1. ACTIVITY SUMMARY

            What is the user working on, reading, or studying?
            What key topics, concepts, or problems are they engaging with?
            What seems to be their current focus or challenge?

        2. PROACTIVE RESEARCH & INSIGHTS

            For identified activities/topics, provide:

            Related concepts: Connected ideas they should know about
            Deeper research: Latest developments, key papers, or advanced topics
            Practical applications: How this knowledge can be applied
            Interesting connections: Cross-disciplinary insights or unexpected angles
            Expert perspectives: What leading practitioners/researchers are saying

        OUTPUT FORMAT
        SUMMARY
            Brief overview of what the user was working on or learning.
        
        RESEARCH & INSIGHTS
            For each topic/activity identified provide additional helpful insights/important information for the user:
            Topic: What they were engaged with
            Key Insights: Relevant research, developments, or expert knowledge
            Connections: How this relates to other fields or concepts they care about
            Next Level: Advanced concepts or applications to explore
            Interesting Angles: Unexpected perspectives or emerging trends
        """
    ]
)

print(response.text)