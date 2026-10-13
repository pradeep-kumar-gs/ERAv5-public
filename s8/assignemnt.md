This one is for the whole cohort, and if it comes out well I want to keep the best one and put it in front of next year's batch.

Work with your AI agent and build a web app, on Netlify or Vercel or wherever you like, that shows every attention mechanism we covered today, visually. Start with the standard one, the plain scaled dot product with a softmax from Session 2, because nothing after it makes sense without it. Then go through all of them.

Here is the part I actually care about, and it is not the animations.

Put them in the order they were launched. Not the order I taught them in, and not grouped by family. Chronological, by the date each one actually appeared. Then explain each one as an answer to a problem that existed at that moment. Vanilla attention was not wrong, it was expensive, and every single thing that follows is somebody looking at that bill and trying to pay less of it. When you lay them out on a timeline you can watch the field change its mind: first it wants exactness, then it wants memory back, then it wants length, then it wants memory back again. You cannot see that from a list. You can see it from a timeline, and once you see it you can guess what comes next, which is the whole reason I am asking.

For each mechanism I want pros and cons, honestly written. Not marketing. Every one of these is a trade and every one of them costs something. If you write down a technique with only pros, you have not understood it yet. Say what it buys, say what it gives up, and say when you would actually pick it. A mechanism that is right for a 2K chatbot and wrong for a 1M agent is not a bad mechanism, and your app should be able to say so.

At minimum cover: standard attention, absolute learned positions, sinusoidal, RoPE, ALiBi, MQA, GQA, sliding window, attention sinks, NTK-aware scaling, YaRN, linear attention, the delta rule and Gated DeltaNet, MLA, sparse and top-k attention, compressed and sparse attention as DeepSeek does it, and DroPE. Add anything I missed. If you find something I did not cover, that is a point in your favour, not a problem.

Make it something you would actually send to a friend who asked "how does attention work now". Interactive if you can manage it, animated if it helps, but a clear static page beats a broken clever one. Your agent can build most of this. Your job is to be right about the dates, right about the trade-offs, and clear about the story.

Submission

Share the live link, plus the GitHub repo behind it. In the README say which sources you used for the dates, because that is the part that is easiest to get wrong and easiest to check.

One warning. Your agent will happily invent a launch date and describe a technique it has half remembered. Check every date against the actual paper or release. I got a mechanism wrong in Session 7 by trusting a confident sentence, and I had to correct it in front of all of you at the top of today's class. Do not repeat my mistake, and if you catch me in another one, tell me.