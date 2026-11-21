"""Shared, evidence-driven farm/shop workflows. Missing evidence always stops input."""
import contextlib
import subprocess
import json
import time
from pathlib import Path
import cv2
import numpy as np
from bot_config import CONFIG, resolve

class Stopped(RuntimeError):
    pass

class EvidenceError(RuntimeError):
    pass

class Vision:
    def __init__(self, bot):
        self.bot = bot

    def matches(self, name, image, roi=None):
        path = resolve(CONFIG['templates_dir']) / (name + '.png')
        template = cv2.imread(str(path)) if path.exists() else None
        if template is None:
            raise EvidenceError('missing_template:' + name)
        h, w = image.shape[:2]
        bounds = roi or CONFIG['rois'].get(name, [0, 0, 1, 1])
        x0, y0, x1, y1 = [int(v * n) for v, n in zip(bounds, (w, h, w, h))]
        crop = image[y0:y1, x0:x1]
        th, tw = template.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            return []
        scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        found = []
        while len(found) < 100:
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < CONFIG['match_threshold']:
                break
            found.append((x0+x+tw//2, y0+y+th//2, float(score)))
            scores[max(0,y-th//2):y+th//2+1, max(0,x-tw//2):x+tw//2+1] = -1
        return found

    def point(self, name, image, roi=None):
        matches = self.matches(name, image, roi)
        if not matches:
            raise EvidenceError('unconfirmed:' + name)
        return matches[0][:2]

    def state(self, name, image):
        return bool(self.matches(name, image))

    def camera(self, before, after, excluded=()):
        if before.shape != after.shape:
            raise EvidenceError('screen_size_changed')
        gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        h, w = gray.shape
        mask[int(h*.12):int(h*.88), int(w*.04):int(w*.96)] = 255
        for x0,y0,x1,y1 in excluded:
            mask[max(0,y0-20):y1+20, max(0,x0-20):x1+20] = 0
        pts = cv2.goodFeaturesToTrack(gray, 150, .03, 12, mask=mask)
        if pts is None or len(pts) < 8:
            raise EvidenceError('missing_camera_landmarks')
        target = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray, target, pts, None)
        if nxt is None:
            raise EvidenceError('lost_camera_landmarks')
        a, b = pts[status.ravel()==1], nxt[status.ravel()==1]
        if len(a) < 8:
            raise EvidenceError('lost_camera_landmarks')
        transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
        if transform is None or int(inliers.sum()) < 8:
            raise EvidenceError('inconsistent_camera_landmarks')
        if abs(transform[0,0]-1) > .015 or abs(transform[0,1]) > .015:
            raise EvidenceError('camera_zoom_or_rotation')
        return float(transform[0,2]), float(transform[1,2])

class Workflow:
    def __init__(self, bot):
        self.b = bot
        self.v = Vision(bot)
        self.reason = 'ready'
        self.debug = resolve(CONFIG['debug_dir'])
        self.debug.mkdir(parents=True, exist_ok=True)

    def log(self, action, **details):
        record = dict(time=time.time(), action=action, device=self.b.device, **details)
        with (self.debug/'steps.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=False))

    def check(self):
        self.b.check_stop()

    def stable(self):
        previous = self.b.screenshot()
        for _ in range(16):
            self.check()
            time.sleep(.2)
            current = self.b.screenshot()
            if previous.shape == current.shape and np.mean(cv2.absdiff(previous,current)) <= CONFIG['stable_threshold']:
                return current
            previous = current
        raise EvidenceError('screen_not_stable')

    def tap(self, point, source):
        self.check()
        self.log('tap', coordinates=point, source=source, before=source)
        self.b.tap(*point)

    def fail(self, error, before=None):
        self.reason = str(error)
        stamp = str(time.time_ns())
        if before is not None:
            cv2.imwrite(str(self.debug/(stamp+'_before.png')), before)
        try:
            cv2.imwrite(str(self.debug/(stamp+'_after.png')), self.b.screenshot())
        except Exception:
            pass
        self.log('stop', reason=self.reason)

class FarmWorkflow(Workflow):
    def run(self, plant=False, zones=None):
        count = 0
        before = None
        try:
            with self.b.task():
                for _ in range(300):
                    self.check()
                    before = self.stable()
                    self.v.point('farm_screen', before)
                    field = self.v.matches('field_empty' if plant else 'field_ripe', before)
                    if zones:
                        field = [p for p in field if any(x0<=p[0]<=x1 and y0<=p[1]<=y1 for x0,y0,x1,y1 in zones)]
                    if not field:
                        self.reason = 'complete' if count else 'no_confirmed_fields'
                        break
                    x,y,_ = sorted(field, key=lambda p:(p[1],p[0]))[0]
                    excluded = [(p[0]-50,p[1]-40,p[0]+50,p[1]+40) for p in field]
                    self.tap((x,y), 'field_empty' if plant else 'field_ripe')
                    menu = self.stable()
                    mdx, mdy = self.v.camera(before, menu, excluded)
                    if max(abs(mdx),abs(mdy)) > CONFIG['camera_threshold']:
                        raise EvidenceError('camera_moved_opening_tool')
                    if plant:
                        self.v.point('seed_menu', menu)
                        seed = self.v.point('crop_menu_wheat', menu)
                        self.tap(seed, 'seed_menu/wheat')
                    else:
                        tool = self.v.point('harvest_tool', menu)
                        self.check()
                        self.log('harvest_drag', coordinates=[tool,(x,y)], source='harvest_tool/field_ripe')
                        self.b.swipe(*tool,x,y, duration_ms=700)
                    after = self.stable()
                    self.v.point('farm_screen', after)
                    dx,dy = self.v.camera(before, after, excluded)
                    self.log('camera', displacement=[dx,dy])
                    h,w = after.shape[:2]
                    tx,ty = x+dx,y+dy
                    roi = (max(0,(tx-25)/w),max(0,(ty-25)/h),min(1,(tx+25)/w),min(1,(ty+25)/h))
                    confirmed = self.v.matches('field_growing' if plant else 'field_empty', after, roi)
                    if not confirmed:
                        raise EvidenceError('target_state_unconfirmed')
                    count += 1
                    self.log('plant' if plant else 'harvest', coordinates=[tx,ty], source='fresh_template', before='empty' if plant else 'ripe', after='growing' if plant else 'empty', confirmed=count)
                    if max(abs(dx),abs(dy)) > CONFIG['camera_threshold']:
                        # User zones are tied to the old frame. Stop rather than reuse them.
                        if zones:
                            raise EvidenceError('camera_moved_reselect_zones')
                        self.log('redetect', reason='camera_moved')
                else:
                    raise EvidenceError('operation_limit')
        except (EvidenceError, Stopped, RuntimeError, OSError, subprocess.SubprocessError) as error:
            self.fail(error, before)
        self.b.last_reason = self.reason
        return count

class VerifiedShopSeller(Workflow):
    def __init__(self, bot, pos):
        super().__init__(bot)
        self.pos = pos
        self.slot = None

    def open_shop(self):
        image = self.stable()
        self.v.point('farm_screen', image)
        candidates = self.v.matches('shop_stand', image)
        self.log('shop_candidates', candidates=candidates, source='fresh_striped_roof_template')
        if not candidates:
            raise EvidenceError('shop_not_found')
        if len(candidates)>1 and candidates[0][2]-candidates[1][2]<.03:
            raise EvidenceError('ambiguous_shop')
        self.log('shop_selected', coordinates=candidates[0][:2], reason='highest_template_score_with_unambiguous_margin')
        self.tap(candidates[0][:2], 'fresh_shop_stand')
        self.v.point('shop_screen', self.stable())
        return True

    def tap_empty_slot(self):
        image = self.stable()
        self.v.point('shop_screen',image)
        slots = self.v.matches('shop_slot_empty',image)
        if not slots:
            # A full shop needs its own positive evidence.
            self.v.point('shop_full',image)
            raise EvidenceError('shop_full')
        self.slot = slots[0][:2]
        self.tap(self.slot,'shop_slot_empty')
        self.v.point('inventory_screen',self.stable())
        return True

    def select_wheat(self):
        image = self.stable()
        self.v.point('inventory_screen',image)
        wheat = self.v.matches('inventory_wheat',image)
        if not wheat:
            self.v.point('out_of_wheat',image)
            raise EvidenceError('out_of_wheat')
        self.tap(wheat[0][:2],'inventory_wheat')
        self.v.point('wheat_selected',self.stable())
        return True

    def set_max_price(self):
        image = self.stable()
        self.v.point('wheat_selected',image)
        if not self.v.state('price_max_reached',image):
            self.tap(self.v.point('btn_price_max',image),'btn_price_max')
        self.v.point('price_max_reached',self.stable())
        return True

    def confirm_sale(self):
        image = self.stable()
        self.v.point('wheat_selected',image)
        self.v.point('price_max_reached',image)
        self.tap(self.v.point('btn_put_sale',image),'btn_put_sale')
        # Never repeat an uncertain submission. Reinspect the exact original slot.
        for _ in range(3):
            image = self.stable()
            if self.v.state('shop_screen',image):
                h,w = image.shape[:2]
                x,y = self.slot
                roi = (max(0,(x-60)/w),max(0,(y-55)/h),min(1,(x+60)/w),min(1,(y+55)/h))
                if self.v.matches('shop_slot_wheat', image, roi):
                    self.log('sale_confirmed', coordinates=self.slot, source='same_slot_wheat', before='empty', after='listed_wheat')
                    return True
        raise EvidenceError('sale_unconfirmed')

    def run(self, slots=1):
        sold = 0
        try:
            if slots < 1:
                raise ValueError('slots must be positive')
            with self.b.task():
                self.open_shop()
                for _ in range(slots):
                    self.check()
                    self.tap_empty_slot()
                    self.select_wheat()
                    self.set_max_price()
                    self.confirm_sale()
                    sold += 1
                self.reason = 'complete'
                self.b.back()
                self.v.point('farm_screen',self.stable())
        except (EvidenceError, Stopped, RuntimeError, OSError, subprocess.SubprocessError) as error:
            self.fail(error)
        self.b.last_reason = self.reason
        self.log('sell_result', slots=sold, reason=self.reason)
        return sold
