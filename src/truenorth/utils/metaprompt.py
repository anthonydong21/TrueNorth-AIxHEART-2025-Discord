# metaprompt.py - shared phrases used in agent prompts

# Design principles
DP1 = "engineer an emotionally intelligent and stereotype-neutral interface within the conversational agent to encourage contextually relevant and engaging interactions, allowing the user to develop high competency and reputation in professional and academic STEM environments"
DP2 = "develop a trustworthy and personalized learning environment through the conversational agent that fosters a sense of local and virtual community among users and fellow STEM colleagues, fostering a sense of community among users to support individualized and high-achieving educational and professional experiences."
DP3 = "facilitate empowering and streamlined interactions with the conversational agent through simplified dialogues to bolster comprehension and foster independence for users in professional and academic STEM environments."

goals_as_str = "\n".join([f"{i}. {goal}" for i, goal in enumerate([DP1, DP2, DP3])])

vectorstore_content_summary = "workplace wellbeing, communcation strategies rooted in conflict resolution and diplomacy, positive psychology, leadership skills, coping mechanisms"

system_relevant_scope = "anything related to optimizing comfort in lived environment, maintaining positive trajectory towards maximizing STEM career"
