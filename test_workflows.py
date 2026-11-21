"""Offline regression checks; never connect to or send input to BlueStacks."""
import contextlib
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from verified_workflow import Vision, EvidenceError, VerifiedShopSeller, FarmWorkflow, CalibratedFarm, Stopped
from bot import BlueStacksBot

class Bot:
    device = 'test'
    def __init__(self):
        self.stop_event = threading.Event()
        self.taps = []
        self.phase = 'farm_screen'
        self.slots = 0
        self.last_reason = ''
        self.fail_sale = False
        self.full = False
        self.no_wheat = False
    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped('stopped')
    @contextlib.contextmanager
    def task(self):
        self.check_stop()
        yield
    def screenshot(self):
        return np.zeros((200,300,3), np.uint8)
    def tap(self,x,y):
        self.check_stop()
        self.taps.append((x,y))
        self.phase = {(10,10):'shop_screen',(50,50):'inventory_screen',(150,50):'inventory_screen',(20,20):'wheat_selected',(30,30):'max',(40,40):'shop_screen',(45,45):'shop_screen',(46,46):'farm_screen'}[(x,y)]
        if (x,y)==(40,40) and not self.fail_sale:
            self.slots+=1
    def back(self):
        self.check_stop()
        self.phase = 'shop_screen' if self.phase in ('inventory_screen', 'wheat_selected', 'max') else 'farm_screen'

class SemanticVision:
    def __init__(self,b): self.b=b
    def require(self, names):
        pass
    def matches(self,name,image,roi=None):
        b=self.b
        if name=='shop_stand': return [(10,10,.99)]
        if name=='shop_slot_empty': return [] if b.full else [(50+100*b.slots,50,.99)]
        if name=='shop_full': return [(1,1,.99)] if b.full else []
        if name=='inventory_wheat': return [] if b.no_wheat else [(20,20,.99)]
        if name=='out_of_wheat': return [(1,1,.99)] if b.no_wheat else []
        if name=='inventory_close': return [(45,45,.99)]
        if name=='shop_close': return [(46,46,.99)]
        if name=='btn_price_max': return [(30,30,.99)]
        if name=='btn_put_sale': return [(40,40,.99)]
        if name=='shop_slot_wheat':
            if b.slots and roi is not None:
                # Require the requested region to contain the slot just submitted.
                expected=50+100*(b.slots-1)
                return [(expected,50,.99)] if roi[0]*300<=expected<=roi[2]*300 else []
            return []
        if name=='wheat_selected': return [(1,1,.99)] if b.phase in ('wheat_selected','max') else []
        if name=='price_max_reached': return [(1,1,.99)] if b.phase=='max' else []
        return [(1,1,.99)] if name==b.phase else []
    def point(self,name,image,roi=None):
        matches=self.matches(name,image,roi)
        if not matches: raise EvidenceError('unconfirmed:'+name)
        return matches[0][:2]
    def state(self,name,image): return bool(self.matches(name,image))

class Regression(unittest.TestCase):
    def seller(self,b):
        s=VerifiedShopSeller(b,None)
        s.v=SemanticVision(b)
        s.stable=b.screenshot
        s.log=lambda *a,**k:None
        s.fail=lambda error,*a:setattr(s,'reason',str(error))
        return s
    def test_two_slots(self):
        b=Bot();s=self.seller(b)
        self.assertEqual(s.run(2),2)
        self.assertEqual(s.reason,'complete')
        self.assertEqual(b.taps.count((40,40)),2)
    def test_unconfirmed_submission_not_repeated(self):
        b=Bot();b.fail_sale=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'sale_unconfirmed')
        self.assertEqual(b.taps.count((40,40)),1)
    def test_full_shop(self):
        b=Bot();b.full=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'shop_full')
        self.assertEqual(b.phase,'farm_screen')
    def test_no_wheat(self):
        b=Bot();b.no_wheat=True;s=self.seller(b)
        self.assertEqual(s.run(2),0)
        self.assertEqual(s.reason,'out_of_wheat')
    def test_unknown_wheat_in_inventory_returns_to_farm(self):
        b = Bot()
        b.no_wheat = True
        seller = self.seller(b)
        with patch('shutil.which', return_value='tesseract.exe'), patch.dict(
                'verified_workflow.CONFIG', {'rois': {'inventory_wheat_count': [],
                                                      'sale_wheat_quantity': []}}):
            self.assertEqual(seller.run(reserve=20), 0)
        self.assertEqual(seller.reason, 'stock_unconfirmed:inventory_wheat')
        self.assertEqual(b.phase, 'farm_screen')
        self.assertNotIn((40, 40), b.taps)

    def test_stop_prevents_next_tap(self):
        b=Bot();s=self.seller(b)
        old=s.select_wheat
        def stop():
            old();b.stop_event.set()
        s.select_wheat=stop
        self.assertEqual(s.run(2),0)
        self.assertNotIn((30,30),b.taps)
        self.assertNotIn((40,40),b.taps)
    def test_stability_ignores_local_animation(self):
        bot = Bot()
        before = np.zeros((192, 312, 3), np.uint8)
        after = before.copy()
        after[40:120, 80:150] = 100
        frames = iter((before, after))
        bot.screenshot = lambda: next(frames)
        flow = FarmWorkflow(bot)
        with patch('verified_workflow.time.sleep'):
            self.assertIs(flow.stable(), after)

    def test_camera_translation(self):
        rng=np.random.default_rng(42)
        before=rng.integers(0,256,(400,600,3),dtype=np.uint8)
        after=cv2.warpAffine(before,np.float32([[1,0,9],[0,1,-6]]),(600,400))
        dx,dy=Vision(None).camera(before,after,[(200,100,400,300)])
        self.assertAlmostEqual(dx,9,delta=.8)
        self.assertAlmostEqual(dy,-6,delta=.8)
    def test_large_camera_pan_excludes_static_hud(self):
        rng = np.random.default_rng(6)
        before = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        after = cv2.warpAffine(before, np.float32([[1, 0, 0], [0, 1, 110]]), (600, 400))
        after[:, :50] = before[:, :50]
        after[:, -50:] = before[:, -50:]
        self.assertEqual(Vision(None).camera(before, after), (0, 110))

    def test_missing_landmarks(self):
        image=np.zeros((400,600,3),np.uint8)
        with self.assertRaises(EvidenceError): Vision(None).camera(image,image)
    def test_missing_template_stops_without_input(self):
        b=Bot();s=self.seller(b);s.v=Vision(b)
        with tempfile.TemporaryDirectory() as directory, patch.dict('verified_workflow.CONFIG',{'templates_dir':directory}):
            self.assertEqual(s.run(),0)
        self.assertEqual(b.taps,[])
        self.assertIn('farm_screen.png', s.reason)
        self.assertIn('shop_slot_wheat.png', s.reason)
    def test_adb_failure_propagates(self):
        import subprocess
        b=BlueStacksBot()
        b.device = 'offline-adb-error-test'
        with patch('bot.subprocess.run',side_effect=subprocess.CalledProcessError(1,['adb'])):
            with self.assertRaises(subprocess.CalledProcessError): b.tap(1,2)
    def farm(self, *, plant=False, move=0, manual=False):
        b = Bot()
        b.phase = 'field_empty' if plant else 'field_ripe'
        b.drags = []
        previous = b.phase
        result = 'field_growing' if plant else 'field_empty'

        def tap(x, y):
            b.taps.append((x, y))
            b.phase = 'tool'

        def drag(*coords):
            b.drags.append(coords)
            b.phase = result

        class FieldVision:
            def drag_point(self, name, center):
                return center

            def tool_open(self, image):
                return False

            def require(self, names):
                pass

            def point(self, name, image):
                if name in ('seed_menu', 'crop_menu_wheat', 'harvest_tool'):
                    if b.phase != 'tool':
                        raise EvidenceError('unconfirmed:' + name)
                return (20, 30)

            def matches(self, name, image, roi=None):
                if b.phase == previous and name == previous:
                    return [(50, 50, .99)]
                if b.phase == 'tool' and name == previous:
                    return [(50+move, 50, .99)]
                if b.phase == result and name == result:
                    return [(50+move, 50, .99)]
                return []

            def camera(self, before, after, excluded):
                return (move, 0) if b.phase == 'tool' else (0, 0)

        b.tap, b.drag = tap, drag
        f = FarmWorkflow(b, zones=[(0, 0, 100, 100)] if manual else None)
        if not manual:
            f.zones = [(0, 0, 100, 100)]  # A saved calibration was loaded.
        f.stable = b.screenshot
        f.log = lambda *a, **k: None
        f.fail = lambda error, *a: setattr(f, 'reason', str(error))
        f.v = FieldVision()
        return b, f

    def test_plant_drags_seed_to_same_empty_cell(self):
        b, f = self.farm(plant=True)
        self.assertEqual(f.run(plant=True, limit=1), 1)
        self.assertEqual(b.taps, [(50, 50)])
        self.assertEqual(b.drags, [(20, 30, 50, 50)])

    def test_harvest_waits_for_transient_bundle_without_dragging_again(self):
        b, f = self.farm()
        matches = f.v.matches
        observations = []

        def without_first_empty(name, image, roi=None):
            if name == b.phase == 'field_empty':
                observations.append(name)
                if len(observations) == 1:
                    return []
            return matches(name, image, roi)

        f.v.matches = without_first_empty
        self.assertEqual(f.run(limit=1), 1)
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(b.drags), 1)

    def test_camera_shift_rescans_before_harvest(self):
        b, f = self.farm(move=10)
        self.assertEqual(f.run(limit=1), 1)
        self.assertEqual(b.drags, [(20, 30, 60, 50)])

    def test_camera_drift_stops_user_zones(self):
        b, f = self.farm(move=10, manual=True)
        self.assertEqual(f.run(), 0)
        self.assertTrue(f.reason.startswith('camera_moved_reselect_zones'))
        self.assertEqual(b.drags, [])

    def test_missing_result_template_prevents_opening_tool(self):
        b, f = self.farm(plant=True)
        f.v = Vision(b)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'verified_workflow.CONFIG', {'templates_dir': directory}):
            for name in ('farm_screen', 'field_empty', 'seed_menu', 'crop_menu_wheat', 'field_selected_empty'):
                image = np.arange(75, dtype=np.uint8).reshape((5, 5, 3))
                cv2.imwrite(directory + '/' + name + '.png', image)
            self.assertEqual(f.run(plant=True), 0)
        self.assertIn('field_growing.png', f.reason)
        self.assertFalse(b.taps or b.drags)

    def test_sale_limits_quantity_to_confirmed_surplus(self):
        b = Bot()
        b.phase = 'inventory_screen'
        s = self.seller(b)
        numbers = iter([18, 10, 9, 8])
        s.confirmed_number = lambda *a: next(numbers)
        old_point = s.v.point
        s.v.point = lambda name, image: (25, 25) if name == 'btn_quantity_minus' else old_point(name, image)
        tap = b.tap
        b.tap = lambda x, y: b.taps.append((x, y)) if (x, y) == (25, 25) else tap(x, y)
        self.assertEqual(s.select_surplus(10), 8)
        self.assertEqual(b.taps.count((25, 25)), 2)

class CalibratedFarmRegression(unittest.TestCase):
    def test_relative_anchor_uses_template_top_left(self):
        vision = Vision(None)
        vision.templates['harvest_tool'] = np.zeros((21, 41, 3), np.uint8)
        with patch.dict('verified_workflow.CONFIG', {'drag_anchors': {'harvest_tool': [.25, .7]}}):
            self.assertEqual(vision.drag_point('harvest_tool', (150, 80)), (140, 84))

    def test_dense_shoot_texture_differs_from_plain_grass(self):
        hsv = np.zeros((64,64,3), np.uint8)
        hsv[:] = (40,190,170)
        grass = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hsv[:, ::2, 2] = 105
        shoots = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(CalibratedFarm.plot_state(grass, (32,32)), 'unknown')
        self.assertEqual(CalibratedFarm.plot_state(shoots, (32,32)), 'growing')

    def make_farm(self):
        bot = Bot()
        farm = CalibratedFarm(bot)
        states = ['ripe'] * 5 + ['empty']
        attempts = []
        markers = []
        farm.load_plots = lambda: None
        farm.stable = bot.screenshot
        farm.v.require = lambda names: None
        farm.v.tool_open = lambda frame: False
        farm.log = lambda *args, **kwargs: None
        farm.scan = lambda *a, **kw: [((50+35*i, 50), value) for i, value in enumerate(states)]
        farm.evidence = lambda reason, i, *_: markers.append((i, reason))

        def open_tool(index, cells, planting):
            attempts.append((index, planting))
            if index != 2:
                states[index] = 'growing' if planting else 'empty'
            return bot.screenshot()

        farm.open_tool = open_tool
        return farm, states, attempts, markers

    def test_middle_plot_retries_then_continues_and_replants_only_confirmed(self):
        farm, states, attempts, markers = self.make_farm()
        result = farm.harvest_and_plant()
        self.assertEqual(result['reason'], 'complete_with_remaining')
        self.assertEqual((result['harvested'], result['planted'], result['remaining']), (4, 4, 1))
        self.assertEqual(result['remaining_plots'], [3])
        self.assertEqual(attempts, [(0,False), (1,False), (2,False), (2,False), (3,False), (4,False),
                                    (0,True), (1,True), (3,True), (4,True)])
        self.assertEqual(markers, [(2,'transition_ripe')])
        self.assertEqual(states, ['growing','growing','ripe','growing','growing','empty'])

    def test_transient_hud_obscured_by_reward_rechecks_without_input(self):
        farm, _, attempts, _ = self.make_farm()
        old_scan = farm.scan
        obscured = False

        def scan(*args, **kwargs):
            nonlocal obscured
            if attempts and not obscured:
                obscured = True
                raise EvidenceError('unconfirmed:farm_screen')
            return old_scan(*args, **kwargs)

        farm.scan = scan
        result = farm.harvest_and_plant(limit=1)
        self.assertEqual((result['harvested'], result['planted']), (1, 1))
        self.assertEqual(attempts, [(0, False), (0, True)])

    def test_unknown_screen_stops_before_next_plot(self):
        farm, states, attempts, _ = self.make_farm()
        old_scan = farm.scan

        def scan(*args, **kwargs):
            if attempts:
                raise EvidenceError('lost_camera_landmarks')
            return old_scan(*args, **kwargs)

        farm.scan = scan
        result = farm.harvest_and_plant()
        self.assertEqual(result['reason'], 'lost_camera_landmarks')
        self.assertEqual(attempts, [(0,False)])
        self.assertEqual(result['harvested'], 0)

class Integration(unittest.TestCase):
    def test_menu_8_through_13(self):
        import run
        for choice in range(8,14):
            answers = [str(choice)] + (["2"] if choice in (8,11) else []) + ["0"]
            with self.subTest(choice=choice), patch("builtins.input",side_effect=answers), patch("builtins.print"), patch("run.Positions") as positions, patch("run.ShopSeller") as seller, patch("run.UpgradeManager") as upgrade, patch("run.calibrate",return_value="farm.png"), patch("run.calibrate_shop",return_value="shop.png"), patch("subprocess.Popen") as launch:
                bot=object()
                run.menu(bot)
                if choice in (8,11):
                    seller.assert_called_once_with(bot,positions.return_value)
                    seller.return_value.run.assert_called_once_with(slots=2)
                if choice in (9,10,11):
                    upgrade.assert_called_once_with(bot,positions.return_value)
                if choice in (12,13):
                    launch.assert_called_once_with(["explorer","farm.png" if choice==12 else "shop.png"])
    def test_template_crop_uses_the_previewed_android_frame(self):
        from hayday import HayDayBot
        from pathlib import Path
        bot = HayDayBot()
        image = np.arange(100*160*3, dtype=np.uint8).reshape((100, 160, 3))
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                'bot_config.CONFIG', {'templates_dir': directory}), patch.object(
                bot, 'screenshot', return_value=image) as screenshot, patch(
                'hayday.cv2.selectROI', return_value=(5, 8, 14, 12)), patch(
                'hayday.cv2.destroyAllWindows'), patch('builtins.print'), patch(
                'builtins.input', return_value='field_ripe.png'):
            bot.capture_template_interactive()
            saved = cv2.imread(str(Path(directory) / 'field_ripe.png'))
        screenshot.assert_called_once()
        np.testing.assert_array_equal(saved, image[16:40, 10:38])

    def test_unknown_stock_skips_without_adb_input(self):
        b = Bot()
        seller = Regression().seller(b)
        with patch('shutil.which', return_value=None):
            self.assertEqual(seller.run(reserve=20), 0)
        self.assertIn('stock_unconfirmed', seller.reason)
        self.assertEqual(b.taps, [])

    def test_device_lock_excludes_other_thread(self):
        first=BlueStacksBot();second=BlueStacksBot();results=[]
        first.device = second.device = 'offline-device-lock-test'
        def concurrent():
            try:
                with second.task(): results.append("entered")
            except RuntimeError: results.append("busy")
        with first.task():
            t=threading.Thread(target=concurrent);t.start();t.join(3)
        self.assertEqual(results,["busy"])
        with second.task(): pass

if __name__=='__main__': unittest.main()
