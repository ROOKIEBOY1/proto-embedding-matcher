"""Create synthetic fixtures BEFORE inference; labels derive only from this specification."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# enum, field, prefix, (suffix, wire ID, canonical wording, synonym)
SPECS = [
 ('CameraModeType','Camera Mode','CAMERA_MODE_TYPE',[
  ('SLOWMOTION',0,'Slow Motion','slow-motion video'),('VIDEONORMAL',1,'Normal Video','standard video'),
  ('TIMELAPSE',2,'Time Lapse','time-lapse video'),('PHOTO',3,'Photo','still photography'),
  ('BURST',4,'Burst','continuous photo shooting'),('NIGHT',5,'Night','night photography')]),
 ('WorkingModeType','Working Mode','WORKING_MODE_TYPE',[
  ('CAPTURE',0,'Capture','image capture'),('RECORDING',1,'Recording','record video'),
  ('PLAYBACK',2,'Playback','play recorded media'),('TRANSCODE',3,'Transcode','convert video encoding')]),
 ('VideoFrameRateType','Video Frame Rate','VIDEO_FRAME_RATE_TYPE_',[
  ('24FPS',1,'24 FPS','24 frames per second'),('30FPS',2,'30 FPS','30 frames per second'),
  ('48FPS',4,'48 FPS','48 frames per second'),('60FPS',6,'60 FPS','60 frames per second'),
  ('120FPS',7,'120 FPS','120 frames per second'),('240FPS',8,'240 FPS','240 frames per second'),
  ('1440FPS',13,'1440 FPS','1440 frames per second')]),
 ('WhiteBalanceType','White Balance','WHITE_BALANCE_TYPE',[
  ('AUTO',0,'Auto','automatic white balance'),('DAYLIGHT',1,'Daylight','sunlight'),
  ('CLOUDY',2,'Cloudy','overcast lighting'),('TUNGSTEN',3,'Tungsten','incandescent lighting'),
  ('FLUORESCENT',4,'Fluorescent','fluorescent lighting')]),
 ('VideoResolutionType','Video Resolution','VIDEO_RESOLUTION_TYPE',[
  ('720P',0,'720 P','720p video'),('1080P',1,'1080 P','1080p video'),
  ('2160P',2,'2160 P','2160p video')]),
 ('VideoCodecType','Video Codec','VIDEO_CODEC_TYPE',[
  ('H264',0,'H264','H.264'),('H265',1,'H265','H.265'),('MJPEG',2,'MJPEG','motion JPEG')]),
 ('StabilizationType','Stabilization','STABILIZATION_TYPE',[
  ('OFF',0,'Off','disabled'),('STANDARD',1,'Standard','normal stabilization'),
  ('BOOST',2,'Boost','enhanced stabilization')]),
 ('SharpnessType','Sharpness','SHARPNESS',[
  ('LOW',0,'Low','gentle edge enhancement'),('MEDIUM',1,'Medium','moderate edge enhancement'),
  ('HIGH',2,'High','strong edge enhancement')]),
 ('ExposureCompensationType','Exposure Compensation','EXPOSURE_COMPENSATION_TYPE',[
  ('MINUS_2EV',0,'-2 EV','-2 exposure value'),('MINUS_1EV',1,'-1 EV','-1 exposure value'),
  ('ZERO_0EV',2,'0 EV','0 exposure value'),('PLUS_1EV',3,'1 EV','+1 exposure value'),
  ('PLUS_2EV',4,'2 EV','+2 exposure value')]),
 ('IsoType','ISO','ISO_TYPE',[
  ('100',0,'100','ISO 100'),('200',1,'200','ISO 200'),('400',2,'400','ISO 400'),('800',3,'800','ISO 800')]),
 ('ImageTextureType','Image Texture','IMAGE_TEXTURE_TYPE',[
  ('SOFT',0,'Soft','soft rendering'),('NATURAL',1,'Natural','natural rendering'),('CRISP',2,'Crisp','crisp rendering')]),
]

ALIASES = {
 'Camera Mode':'Camera shooting mode', 'Working Mode':'Device operation',
 'Video Frame Rate':'Frame Rate', 'White Balance':'Color balance',
 'Video Resolution':'Video image resolution', 'Video Codec':'Video encoding format',
 'Stabilization':'Image stabilization', 'Sharpness':'Edge sharpness',
 'Exposure Compensation':'Exposure adjustment', 'ISO':'Sensor sensitivity',
 'Image Texture':'Image rendering texture',
}


def main():
    """Write immutable-labeled schema and template-derived dev/test rows."""
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    proto = ['syntax = "proto2";', '// Synthetic capability fixture, not a real product schema.',
             '// Numeric enum IDs are wire IDs, NOT physical parameter values.',
             '// SHARPNESS_HIGH means strong edge enhancement in this fixture.']
    rows = []
    for enum, field, prefix, members in SPECS:
        proto.append('enum ' + enum + ' {')
        for suffix, number, canonical, synonym in members:
            name = prefix + '_' + suffix
            proto.append(f'  {name} = {number};')
            category = ('new_enum' if enum == 'ImageTextureType' else
                        'new_member' if suffix == '1440FPS' else
                        'numeric' if any(c.isdigit() for c in canonical) else 'standard_synonym')
            for split, variants in [
                ('dev', [(field, 'Please select ' + canonical),
                         (field, 'Use ' + synonym)]),
                ('test', [(field, canonical), (ALIASES[field], synonym),
                          (field, 'Set it to ' + synonym),
                          (ALIASES[field], 'I would like ' + canonical)])]:
                for i, (f, v) in enumerate(variants):
                    rows.append(dict(id=f'{split}_{name}_{i}', split=split, field=f,
                                     value=v, expected_enum=enum, expected=name,
                                     category=category, source='synthetic_template',
                                     template=f'{split}_{i}', expected_number=number))
        proto.append('}')
    # Controlled negatives. Field is labeled even for unsupported values;
    # entirely unrelated fields have no expected enum.
    negatives = {
     'dev': [
      ('Frame Rate',v,'VideoFrameRateType','unsupported_number') for v in
      ['25 FPS','47 FPS','59 FPS','119 FPS','239 FPS','1441 FPS','300 FPS','1000 FPS']
     ] + [
      ('Camera Mode','underwater panorama','CameraModeType','unsupported_mode'),
      ('Camera Mode','stereoscopic 3D','CameraModeType','unsupported_mode'),
      ('Working Mode','firmware upgrade','WorkingModeType','near_semantic'),
      ('White Balance','custom Kelvin preset','WhiteBalanceType','near_semantic'),
      ('Video Resolution','900 P','VideoResolutionType','unsupported_number'),
      ('ISO','320','IsoType','unsupported_number'),
      ('Exposure Compensation','0.5 EV','ExposureCompensationType','unsupported_number'),
      ('Weather','thunderstorms tomorrow',None,'unrelated'),
      ('Navigation','shortest route home',None,'unrelated'),
      ('Shopping','add apples to basket',None,'unrelated'),
      ('Audio','Dolby surround',None,'unsupported_field'),
      ('GPS','enabled',None,'unsupported_field')],
     'test': [
      ('Frame Rate',v,'VideoFrameRateType','unsupported_number') for v in
      ['144 FPS','23 FPS','29 FPS','49 FPS','61 FPS','121 FPS','241 FPS','1439 FPS',
       '14 FPS','480 FPS','960 FPS','144.0 FPS','59.94 FPS','119.88 FPS','2400 FPS']
     ] + [
      ('Camera Mode',v,'CameraModeType','unsupported_mode') for v in
      ['Hyperlapse','Panorama','underwater photography','thermal imaging','360 degree video']
     ] + [
      ('Working Mode','live streaming','WorkingModeType','near_semantic'),
      ('Working Mode','standby','WorkingModeType','near_semantic'),
      ('White Balance','shade','WhiteBalanceType','near_semantic'),
      ('White Balance','6500 Kelvin','WhiteBalanceType','near_semantic'),
      ('Stabilization','horizon lock','StabilizationType','near_semantic'),
      ('Image Texture','vivid','ImageTextureType','near_semantic'),
      ('Sharpness','extra high','SharpnessType','near_semantic'),
      ('Video Codec','AV1','VideoCodecType','unsupported_codec'),
      ('Video Resolution','1440 P','VideoResolutionType','unsupported_number'),
      ('Video Resolution','4320 P','VideoResolutionType','unsupported_number'),
      ('ISO','160','IsoType','unsupported_number'),
      ('ISO','1600','IsoType','unsupported_number'),
      ('Exposure Compensation','-0.3 EV','ExposureCompensationType','unsupported_number'),
      ('Exposure Compensation','3 EV','ExposureCompensationType','unsupported_number'),
      ('Lunch','please order a pizza',None,'unrelated'),
      ('Finance','what is the exchange rate',None,'unrelated'),
      ('Music','play a jazz song',None,'unrelated'),
      ('Focus','continuous autofocus',None,'unsupported_field'),
      ('Audio Gain','12 dB',None,'unsupported_field'),
      ('GPS Position','enable location tagging',None,'unsupported_field')]
    }
    for split, cases in negatives.items():
        for i, (field, value, enum, category) in enumerate(cases):
            rows.append(dict(id=f'{split}_unknown_{i}',split=split,field=field,value=value,
                             expected_enum=enum,expected=None,expected_number=None,
                             category=category,source='curated_synthetic_negative',template='manual'))
    (data/'camera.proto').write_text('\n'.join(proto)+'\n',encoding='utf-8')
    (data/'dataset.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # Equal semantics asserted by fixture contract before inference, never by predictions.
    (data/'naming_variants.json').write_text(json.dumps({
      'contract':'In these two synthetic schemas only: HIGH and PLUS both mean strong edge enhancement.',
      'schemas':[
       'enum SharpnessType { SHARPNESS_LOW=0; SHARPNESS_MEDIUM=1; SHARPNESS_HIGH=2; }',
       'enum ImageAcutanceType { IMAGE_ACUTANCE_MINUS=0; IMAGE_ACUTANCE_NORMAL=1; IMAGE_ACUTANCE_PLUS=2; }'],
      'queries':['Sharpness: High','Sharpness: strong edge enhancement','Image acutance: increased edge definition'],
      'expected':['SHARPNESS_HIGH','IMAGE_ACUTANCE_PLUS']},indent=2),encoding='utf-8')
    print(f'Frozen {len(rows)} labeled samples, {sum(x["split"]=="test" and x["expected"] is not None for x in rows)} supported test samples.')


if __name__ == '__main__':
    main()
