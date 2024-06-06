import os
import tqdm
import pandas as pd
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api._errors import NoTranscriptFound,TranscriptsDisabled
from deepmultilingualpunctuation import PunctuationModel


load_dotenv()
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
DF_COLUMNS = ["playlist", "title", "transripts_with_punctuation"]

class LoadTranscripts():
    def __init__(self, transcripts_path: str, playlist_ids_dict:dict,
                 standalone_video_ids:dict = {}):
        self.transcripts_path = transcripts_path
        self.playlist_ids_dict = playlist_ids_dict
        self.standalone_video_ids = standalone_video_ids
        self.punct_restore_model =PunctuationModel()
    # Set up the API key and build the YouTube API client

    
    # Call the playlistItems API to get the video IDs for the playlist

    def get_playlists_video_ids(self):
        video_ids = {}
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        for playlistName, playlistId in self.playlist_ids_dict.items():
            next_page_token = None
            while True:
                playlist_response = youtube.playlistItems().list(
                    part=["contentDetails", "snippet"],
                    playlistId=playlistId,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()

                for item in playlist_response['items']:
                    title = item["snippet"]["title"]
                    if title not in video_ids.keys():
                        playlist_title = "DELIMETER".join([playlistName, title])
                        video_ids[playlist_title] = item['contentDetails']['videoId']

                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
        return video_ids
    
    def get_standalone_video_ids(self):
        vid_rows = []
        for vname, vid in  self.standalone_video_ids.items():
            punct_restore_model =PunctuationModel()
            try:
                transcript = YouTubeTranscriptApi.get_transcript(vid, languages=['ru'])
            except NoTranscriptFound:
                pass
            except TranscriptsDisabled:
                print(f"TranscriptsDisabled: Skipping video {vid}")
                pass
            # Extract the transcript text
            transcript_text = ' '.join([x['text'] for x in transcript])

            text =  punct_restore_model.restore_punctuation(transcript_text)
            
            vid_rows.append(["No playlist", vname, text])
        standalone_df = pd.DataFrame(vid_rows, columns = DF_COLUMNS)
        return standalone_df
        
        

    def get_video_transcripts(self, video_ids):
        transcripts_list_punct = []
        for playlist_title, video_id in tqdm(video_ids.items()):
            # Get the transcript for the specified video ID
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru'])
            except NoTranscriptFound:
                continue
            except TranscriptsDisabled:
                print(f"TranscriptsDisabled: Skipping video {video_id}")
                continue

            # Extract the transcript text
            transcript_text = ' '.join([x['text'] for x in transcript])

            playlistName,video_title, = playlist_title.split("DELIMETER")
            transcripts_list_punct.append((playlistName,video_title, self.punct_restore_model.restore_punctuation(transcript_text)))
            
        save_punct_transcripts = "../../data/interim/punctuated_trancripts.csv".format()
        transcripts_df = pd.DataFrame(transcripts_list_punct, columns = DF_COLUMNS)
        transcripts_df.to_csv(save_punct_transcripts, index=False,encoding = "utf-8")   
        return transcripts_df, transcripts_list_punct
    
    def __call__(self):
        #parse videos ids from YT channel playlists
        video_ids = self.get_playlists_video_ids()
        #parse transcripts from videos ids list
        transcripts_df, transcripts_list_punct = self.get_video_transcripts(video_ids)
        return transcripts_df, transcripts_list_punct