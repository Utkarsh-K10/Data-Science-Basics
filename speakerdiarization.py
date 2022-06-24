import os
import sys
import configparser 
import wave
from pydub import AudioSegment
AudioSegment.converter ='/usr/local/Cellar/ffmpeg/4.3.1/bin/ffmpeg'
from datetime import datetime
from google.cloud import storage
from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types


# GLOBAL_IINPUT_AUDIO_LANGAUGE =()
class Transcribe:

    config = configparser.ConfigParser()
    config.read('speech_recog.ini')   
    bucket_name = config['CREDENTIALS']['BUCKET_NAME']
    jsonfile = config['CREDENTIALS']['JSON']
#     language_code = 
    supported = [
        "wav",
        "mp3",
        "ogg",
    ]

    def __init__(self, audiofile):

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.jsonfile

        self.audiofile = audiofile
        self.audioext = self.audiofile.split(".")[1]
        self.wavfile = os.path.basename(self.audiofile.split(".")[0]) + ".wav"
        self.transcriptfile = f"transcript/{os.path.basename(self.audiofile)[0]}" +".txt"
        if not os.path.isdir("transcript"):
            os.mkdir("transcript")
        self.frame_rate = None
        self.channels = None
        if not self.audioext in self.supported:
            raise Exception(f"Unknown Ext: {self.audioext}")
        self.toWav()

    def toWav(self):
        if not os.path.isfile(self.wavfile):
            if self.audioext == "wav":
                return
            elif self.audioext == "mp3":
                sound = AudioSegment.from_mp3(self.audiofile)
            elif self.audioext == "ogg":
                sound = AudioSegment.from_ogg(self.audiofile)
            elif self.audioext == "flac":
                sound = AudioSegment.from_flac(self.audiofile)
            newsound = sound.set_channels(1)
            print(f"Converting {self.audiofile} to {self.wavfile}")
            newsound.export(self.wavfile, format="wav")

        with wave.open(self.wavfile, "rb") as wave_file:
            self.frame_rate = wave_file.getframerate()
            self.channels = wave_file.getnchannels()
            if not 1 == self.channels:
                raise Exception("There can only be one channel in wav file")

    def uploadBlob(self):
        """Uploads a file to the bucket."""
        self.destination_name = self.wavfile
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)
        self.blob = bucket.blob(self.destination_name)
        self.blob.upload_from_filename(self.wavfile)
        os.unlink(self.wavfile)

    def deleteBlob(self):
        """Deletes a file from the bucket."""
        self.blob.delete()

    def transcribeAudio(self):
        self.destination_name = self.wavfile
        self.gcs_uri = f"gs://{self.bucket_name}/{self.destination_name}"

        t0 = datetime.now()
#         client = speech.SpeechClient()
#         audio = types.RecognitionAudio(uri=self.gcs_uri)

#         config = types.RecognitionConfig(
#             encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=self.frame_rate,
#             language_code="te-IN")

        # Detects speech in the audio file
#         operation = client.long_running_recognize(config, audio)
#         response = operation.result(timeout=10000)
        t1 = datetime.now()

#         time_taken = t1 - t0
#         print(f"Translation time: {time_taken}")

#         print(response)

#         transcript = ""
#         for result in response.results:
#             transcript += result.alternatives[0].transcript

#         print(transcript)
#         f = open(self.transcriptfile, "w")
#         f.writelines(transcript)
#         f.close()
        
        
        
        client = speech_v1p1beta1.SpeechClient()

        # The language of the supplied audio
    
        diaz_config = types.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count = 2,
            max_speaker_count = 6
        )
    
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.frame_rate,
            language_code='te-IN', #this language code should be dyanamic,actutaly this is language spoken in input audio 
            diarization_config = diaz_config,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True
        )
    
        audio = {"uri": self.gcs_uri}

        operation = client.long_running_recognize(config, audio)
        response = operation.result(timeout=10000)

        for result in response.results:
            # First alternative has words tagged with speakers
            alternative = result.alternatives[0]
            #print("Transcript: {}".format(alternative.transcript))
            speakers = {}
            # Print the speaker_tag of each word
            for word in alternative.words:
                if not speakers.get(word.speaker_tag):
                    speakers[word.speaker_tag] = [word.word]
                else:
                    speakers[word.speaker_tag].append(word.word)
        
        t1 = datetime.now()

        time_taken = t1 - t0
        print(f"Translation time: {time_taken}")
        print(speakers)
        f = open(self.transcriptfile, "w")
        f.writelines(speakers)
        f.close()

def main(audiofile):
    t = Transcribe(audiofile)
#     t.uploadBlob()
    t.transcribeAudio()
    t.deleteBlob()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        audiofile = sys.argv[1]
    else:
        audiofile = "marathi-english.mp3"

    main('telugu_noise.wav')