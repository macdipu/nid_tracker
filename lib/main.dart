import 'dart:math' as math;
import 'dart:io';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/rendering.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:image/image.dart' as img;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]).then((_) => runApp(const MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'NID Detector',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  void _openSidePicker(BuildContext context, {required bool boxesOnly}) {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.credit_card),
              title: const Text('Front side'),
              onTap: () {
                Navigator.pop(ctx);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => NidCaptureYoloViewPage(
                      title: boxesOnly ? 'Front - Boxes Only' : 'Front - Capture',
                      modelAssetPath: 'assets/front_nid_model.tflite',
                      labelsAssetPath: 'assets/front_nid_labels.txt',
                      showBoxesOnly: boxesOnly,
                    ),
                  ),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.credit_card_rounded),
              title: const Text('Back side'),
              onTap: () {
                Navigator.pop(ctx);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => NidCaptureYoloViewPage(
                      title: boxesOnly ? 'Back - Boxes Only' : 'Back - Capture',
                      modelAssetPath: 'assets/back_nid_model.tflite',
                      labelsAssetPath: 'assets/back_nid_labels.txt',
                      showBoxesOnly: boxesOnly,
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NID Tracker')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 280,
                child: FilledButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const NidLiveDetectPage(
                          title: 'NID Front - Live Detection',
                          modelAssetPath: 'assets/front_nid_model.tflite',
                          labelsAssetPath: 'assets/front_nid_labels.txt',
                        ),
                      ),
                    );
                  },
                  child: const Text('Open NID Front Detector'),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: 280,
                child: OutlinedButton(
                  onPressed: () {
                    // Example for reusing with a different model/labels.
                    // Update the asset paths once you add the back model files to pubspec assets.
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const NidLiveDetectPage(
                          title: 'NID Back - Live Detection',
                          modelAssetPath: 'assets/back_nid_model.tflite',
                          labelsAssetPath: 'assets/back_nid_labels.txt',
                        ),
                      ),
                    );
                  },
                  child: const Text('Open NID Back Detector'),
                ),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: 280,
                child: FilledButton.tonal(
                  onPressed: () => _openSidePicker(context, boxesOnly: false),
                  child: const Text('Use Camera'),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: 280,
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.view_compact_alt),
                  onPressed: () => _openSidePicker(context, boxesOnly: true),
                  label: const Text('Show Bounding Box'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class NidLiveDetectPage extends StatefulWidget {
  final String title;
  final String modelAssetPath;
  final String labelsAssetPath;
  const NidLiveDetectPage({
    super.key,
    required this.title,
    required this.modelAssetPath,
    required this.labelsAssetPath,
  });

  @override
  State<NidLiveDetectPage> createState() => _NidLiveDetectPageState();
}

class _NidLiveDetectPageState extends State<NidLiveDetectPage> {
  final _controller = YOLOViewController();
  List<YOLOResult> _results = const [];
  Map<String, YOLOResult> _latestByLabel = {};
  double _zoom = 1.0;

  // Absolute model file path copied from assets
  String? _modelFilePath;

  // Labels loaded from assets/labels.txt
  List<String> _labels = const [];
  bool _labelsLoaded = false;
  double? _fps;
  int _lastEventMs = 0;

  @override
  void initState() {
    super.initState();
    _prepareModelPath();
    _loadLabels();
    // Apply more permissive thresholds to ensure early visibility
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.setThresholds(confidenceThreshold: 0.60, iouThreshold: 0.50, numItemsThreshold: 100);
    });
  }

  Future<void> _prepareModelPath() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${dir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }
      final baseName = widget.modelAssetPath.split('/').last;
      final outFile = File('${modelsDir.path}/$baseName');
      // Always copy on first run; overwrite if file missing or size differs
      final data = await rootBundle.load(widget.modelAssetPath);
      if (!await outFile.exists() || (await outFile.length()) != data.lengthInBytes) {
        await outFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      if (mounted) setState(() => _modelFilePath = outFile.path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Model copy failed: $e')),
      );
    }
  }

  Future<void> _loadLabels() async {
    try {
      final txt = await rootBundle.loadString(widget.labelsAssetPath);
      final lines = txt.split(RegExp(r'\r?\n')).where((l) => l.trim().isNotEmpty).toList();
      if (mounted) {
        setState(() {
          _labels = lines;
          _labelsLoaded = true;
          // Recompute latest detections mapped by label using current _results
          _latestByLabel = {
            for (final r in _results)
              if (_labels.contains(_displayName(r))) _displayName(r): r,
          };
        });
      }
    } catch (_) {
      // ignore if labels not present
    }
  }

  String _displayName(YOLOResult r) {
    final name = r.className.trim();
    final looksNumeric = RegExp(r'^\d+$').hasMatch(name);
    if (_labelsLoaded && (name.isEmpty || looksNumeric) && r.classIndex >= 0 && r.classIndex < _labels.length) {
      return _labels[r.classIndex];
    }
    return name.isEmpty && r.classIndex >= 0 && r.classIndex < _labels.length
        ? _labels[r.classIndex]
        : (name.isEmpty ? 'class_${r.classIndex}' : name);
  }

  @override
  Widget build(BuildContext context) {
    final ready = _modelFilePath != null; // wait until model file is ready
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch),
            onPressed: () => _controller.switchCamera(),
          ),
        ],
      ),
      body: !ready
          ? const Center(child: Text('Preparing model…'))
          : LayoutBuilder(
        builder: (context, constraints) {
          final screenSize = Size(constraints.maxWidth, constraints.maxHeight);
          return Stack(
            fit: StackFit.expand,
            children: [
              YOLOView(
                modelPath: _modelFilePath!,
                task: YOLOTask.detect,
                controller: _controller,
                // Enable native overlay too, for quicker visual feedback
                showNativeUI: false,
                useGpu: true,
                confidenceThreshold: 0.60,
                iouThreshold: 0.50,
                streamingConfig: const YOLOStreamingConfig.minimal(),
                onResult: (List<YOLOResult> results) {
                  final now = DateTime.now().millisecondsSinceEpoch;
                  // Make a fresh copy to force painter repaint comparisons
                  final listCopy = List<YOLOResult>.from(results);
                  setState(() {
                    _lastEventMs = now;
                    _results = listCopy;
                    _latestByLabel = {
                      for (final r in listCopy)
                        if (_labels.contains(_displayName(r))) _displayName(r): r,
                    };
                  });
                },
                onStreamingData: (Map<String, dynamic> stream) {
                  // Fallback path: parse raw stream when onResult is bypassed
                  try {
                    final now = DateTime.now().millisecondsSinceEpoch;
                    final dets = (stream['detections'] as List?) ?? const [];
                    final parsed = dets.whereType<Map>().map((m) => YOLOResult.fromMap(m)).toList();
                    setState(() {
                      _lastEventMs = now;
                      _fps = (stream['fps'] is num) ? (stream['fps'] as num).toDouble() : _fps;
                      _results = parsed; // already a fresh list
                      _latestByLabel = {
                        for (final r in parsed)
                          if (_labels.contains(_displayName(r))) _displayName(r): r,
                      };
                    });
                  } catch (_) {/*ignore parse errors*/}
                },
                onPerformanceMetrics: (m) {
                  setState(() { _fps = m.fps; _lastEventMs = DateTime.now().millisecondsSinceEpoch; });
                },
                onZoomChanged: (z) => setState(() => _zoom = z),
              ),
              // Our overlay for custom styling/colors and summary chips
              CustomPaint(
                painter: _ResultsPainter(
                  results: _results,
                  screenSize: screenSize,
                  nameFor: _displayName,
                  colorForLabel: _colorForLabel,
                ),
              ),
              // Status banner (FPS / No detections yet)
              Positioned(
                top: 12,
                left: 12,
                right: 12,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _StatusBanner(fps: _fps, lastEventMs: _lastEventMs),
                    const SizedBox(height: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.45),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text('Detections: ${_results.length}', style: const TextStyle(color: Colors.white)),
                    ),
                  ],
                ),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: _buildBottomPanel(),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildBottomPanel() {
    return SafeArea(
      top: false,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
        ),
        child: DefaultTextStyle(
          style: const TextStyle(color: Colors.white, fontSize: 14),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text('Detected fields', style: TextStyle(fontWeight: FontWeight.bold)),
                  const Spacer(),
                  Text('Zoom ${_zoom.toStringAsFixed(1)}x'),
                ],
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final key in _labels)
                    _FieldChip(
                      label: key,
                      value: _latestByLabel[key]?.confidence != null
                          ? '${(_latestByLabel[key]!.confidence * 100).toStringAsFixed(0)}%'
                          : '—',
                      color: _colorForLabel(key),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _colorForLabel(String label) {
    if (_labels.isEmpty) return Colors.grey;
    final idx = _labels.indexOf(label);
    final hue = (idx / _labels.length) * 360.0;
    return HSLColor.fromAHSL(1.0, hue, 0.8, 0.5).toColor();
  }
}

class _ResultsPainter extends CustomPainter {
  final List<YOLOResult> results;
  final Size screenSize;
  final String Function(YOLOResult) nameFor;
  final Color Function(String) colorForLabel;

  _ResultsPainter({
    required this.results,
    required this.screenSize,
    required this.nameFor,
    required this.colorForLabel,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (final r in results) {
      final className = nameFor(r);
      final color = colorForLabel(className);
      paint.color = color;

      final nb = r.normalizedBox;
      final rect = Rect.fromLTWH(
        nb.left * size.width,
        nb.top * size.height,
        nb.width * size.width,
        nb.height * size.height,
      );

      canvas.drawRect(rect, paint);

      final label = '$className ${(r.confidence * 100).toStringAsFixed(0)}%';
      textPainter.text = TextSpan(
        text: label,
        style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
      );
      textPainter.layout();
      final tp = Offset(rect.left, math.max(0, rect.top - textPainter.height - 2));
      final bgRect = Rect.fromLTWH(tp.dx - 2, tp.dy - 2, textPainter.width + 4, textPainter.height + 4);
      final bgPaint = Paint()..color = Colors.black.withValues(alpha: 0.55);
      canvas.drawRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(4)), bgPaint);
      textPainter.paint(canvas, tp);
    }
  }

  @override
  bool shouldRepaint(covariant _ResultsPainter oldDelegate) {
    return oldDelegate.results != results || oldDelegate.screenSize != screenSize || oldDelegate.nameFor != nameFor || oldDelegate.colorForLabel != colorForLabel;
  }
}

class _FieldChip extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _FieldChip({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(color: Colors.white)),
          const SizedBox(width: 8),
          Text(value, style: const TextStyle(color: Colors.white70)),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final double? fps;
  final int lastEventMs;
  const _StatusBanner({required this.fps, required this.lastEventMs});

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final stale = now - lastEventMs > 3000; // No events in >3s
    final text = fps != null && !stale
        ? 'FPS ${fps!.toStringAsFixed(1)}'
        : 'No detections/events yet. Check camera permission and model path/format.';
    return Align(
      alignment: Alignment.topLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(text, style: const TextStyle(color: Colors.white)),
      ),
    );
  }
}

// New: Capture Page for camera usage and capture logic
class NidCaptureYoloViewPage extends StatefulWidget {
  final String title;
  final String modelAssetPath;
  final String labelsAssetPath;
  final bool showBoxesOnly;
  const NidCaptureYoloViewPage({
    super.key,
    required this.title,
    required this.modelAssetPath,
    required this.labelsAssetPath,
    this.showBoxesOnly = false,
  });

  @override
  State<NidCaptureYoloViewPage> createState() => _NidCaptureYoloViewPageState();
}

class _NidCaptureYoloViewPageState extends State<NidCaptureYoloViewPage> {
  final _controller = YOLOViewController();
  final _boundaryKey = GlobalKey();

  String? _modelFilePath;
  List<String> _labels = const [];
  bool _labelsLoaded = false;

  List<YOLOResult> _results = const [];
  Map<String, YOLOResult> _latestByLabel = {};
  double? _fps;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _prepareModelPath();
    _loadLabels();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.setThresholds(confidenceThreshold: 0.70, iouThreshold: 0.50, numItemsThreshold: 100);
    });
  }

  Future<void> _prepareModelPath() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${dir.path}/models');
      if (!await modelsDir.exists()) await modelsDir.create(recursive: true);
      final baseName = widget.modelAssetPath.split('/').last;
      final outFile = File('${modelsDir.path}/$baseName');
      final data = await rootBundle.load(widget.modelAssetPath);
      if (!await outFile.exists() || (await outFile.length()) != data.lengthInBytes) {
        await outFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      if (mounted) setState(() => _modelFilePath = outFile.path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Model copy failed: $e')));
    }
  }

  Future<void> _loadLabels() async {
    try {
      final txt = await rootBundle.loadString(widget.labelsAssetPath);
      final lines = txt.split(RegExp(r'\r?\n')).where((l) => l.trim().isNotEmpty).toList();
      if (mounted) setState(() { _labels = lines; _labelsLoaded = true; });
    } catch (_) {}
  }

  String _displayName(YOLOResult r) {
    final name = r.className.trim();
    final looksNumeric = RegExp(r'^\d+$').hasMatch(name);
    if (_labelsLoaded && (name.isEmpty || looksNumeric) && r.classIndex >= 0 && r.classIndex < _labels.length) {
      return _labels[r.classIndex];
    }
    return name.isEmpty && r.classIndex >= 0 && r.classIndex < _labels.length
        ? _labels[r.classIndex]
        : (name.isEmpty ? 'class_${r.classIndex}' : name);
  }

  bool get _allLabelsPresent => _labels.isNotEmpty && _labels.every((l) => _latestByLabel.containsKey(l));

  @override
  Widget build(BuildContext context) {
    final ready = _modelFilePath != null;
    return Scaffold(
      appBar: AppBar(title: Text(widget.title), actions: [
        if (!widget.showBoxesOnly)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Center(
              child: Text(
                _allLabelsPresent ? 'Ready' : 'Align card…',
                style: TextStyle(color: _allLabelsPresent ? Colors.green : Colors.orange),
              ),
            ),
          ),
      ]),
      body: !ready
          ? const Center(child: Text('Preparing model…'))
          : LayoutBuilder(builder: (context, constraints) {
        final screenSize = Size(constraints.maxWidth, constraints.maxHeight);
        return Stack(
          fit: StackFit.expand,
          children: [
            RepaintBoundary(
              key: _boundaryKey,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  YOLOView(
                    modelPath: _modelFilePath!,
                    task: YOLOTask.detect,
                    controller: _controller,
                    showNativeUI: false,
                    useGpu: true,
                    confidenceThreshold: 0.60,
                    iouThreshold: 0.50,
                    streamingConfig: const YOLOStreamingConfig.minimal(),
                    onResult: (List<YOLOResult> results) {
                      final listCopy = List<YOLOResult>.from(results);
                      setState(() {
                        _results = listCopy;
                        _latestByLabel = {
                          for (final r in listCopy)
                            if (_labels.contains(_displayName(r))) _displayName(r): r,
                        };
                      });
                    },
                    onStreamingData: (stream) {
                      try {
                        final dets = (stream['detections'] as List?) ?? const [];
                        final parsed = dets.whereType<Map>().map((m) => YOLOResult.fromMap(m)).toList();
                        setState(() {
                          _fps = (stream['fps'] is num) ? (stream['fps'] as num).toDouble() : _fps;
                          _results = parsed;
                          _latestByLabel = {
                            for (final r in parsed)
                              if (_labels.contains(_displayName(r))) _displayName(r): r,
                          };
                        });
                      } catch (_) {}
                    },
                  ),
                  CustomPaint(
                    painter: _BoxesOnlyPainter(
                      results: _results,
                      screenSize: screenSize,
                      nameFor: _displayName,
                      drawLabels: !widget.showBoxesOnly,
                    ),
                  ),
                ],
              ),
            ),
            if (!widget.showBoxesOnly)
              Positioned(
                bottom: 20,
                left: 16,
                right: 16,
                child: SafeArea(
                  top: false,
                  child: Row(
                    children: [
                      Expanded(
                        child: FilledButton(
                          onPressed: (_allLabelsPresent && !_saving) ? _captureAndSave : null,
                          child: _saving ? const Text('Saving…') : const Text('Capture'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.45),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _fps != null ? 'FPS ${_fps!.toStringAsFixed(1)}' : '—',
                          style: const TextStyle(color: Colors.white),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        );
      }),
    );
  }

  Future<void> _captureAndSave() async {
    try {
      setState(() => _saving = true);
      final boundary = _boundaryKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) throw Exception('No boundary');
      final dpr = MediaQuery.of(context).devicePixelRatio;
      final ui.Image image = await boundary.toImage(pixelRatio: dpr);
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) throw Exception('No image bytes');
      Uint8List bytes = byteData.buffer.asUint8List();

      final cardLabel = _labels.firstWhere(
            (l) => l.toLowerCase().contains('_image') || l.toLowerCase().contains('card'),
        orElse: () => '',
      );
      if (cardLabel.isNotEmpty && _latestByLabel[cardLabel] != null) {
        final r = _latestByLabel[cardLabel]!;
        final nb = r.normalizedBox;
        final decoded = img.decodeImage(bytes);
        if (decoded != null) {
          final iw = decoded.width;
          final ih = decoded.height;
          final x = (nb.left * iw).clamp(0, iw - 1).toInt();
          final y = (nb.top * ih).clamp(0, ih - 1).toInt();
          final w = (nb.width * iw).clamp(1, iw - x).toInt();
          final h = (nb.height * ih).clamp(1, ih - y).toInt();
          final cropped = img.copyCrop(decoded, x: x, y: y, width: w, height: h);
          bytes = Uint8List.fromList(img.encodePng(cropped));
        }
      }

      final side = widget.labelsAssetPath.contains('back') ? 'back' : 'front';
      await _showPreviewAndMaybeSave(bytes: bytes, filePrefix: 'nid_$side');
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Capture failed: $e')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  Future<void> _showPreviewAndMaybeSave({required Uint8List bytes, required String filePrefix}) async {
    if (!mounted) return;
    final saved = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black87,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(16))),
      builder: (ctx) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text('Preview', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(bytes, fit: BoxFit.contain),
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: () => Navigator.pop(ctx, false),
                        child: const Text('Cancel'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: FilledButton(
                        onPressed: () => Navigator.pop(ctx, true),
                        child: const Text('Save'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );

    if (saved == true) {
      final dir = await getTemporaryDirectory();
      final out = File('${dir.path}/${filePrefix}_${DateTime.now().millisecondsSinceEpoch}.png');
      await out.writeAsBytes(bytes);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Saved: ${out.path}')));
    }
  }
}

class _BoxesOnlyPainter extends CustomPainter {
  final List<YOLOResult> results;
  final Size screenSize;
  final String Function(YOLOResult) nameFor;
  final bool drawLabels;
  _BoxesOnlyPainter({required this.results, required this.screenSize, required this.nameFor, this.drawLabels = true});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.tealAccent;
    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (final r in results) {
      final nb = r.normalizedBox;
      final rect = Rect.fromLTWH(nb.left * size.width, nb.top * size.height, nb.width * size.width, nb.height * size.height);
      canvas.drawRect(rect, paint);
      if (drawLabels) {
        final label = '${nameFor(r)} ${(r.confidence * 100).toStringAsFixed(0)}%';
        textPainter.text = TextSpan(text: label, style: const TextStyle(color: Colors.white, fontSize: 12));
        textPainter.layout();
        final tp = Offset(rect.left, math.max(0, rect.top - textPainter.height - 2));
        final bgRect = Rect.fromLTWH(tp.dx - 2, tp.dy - 2, textPainter.width + 4, textPainter.height + 4);
        final bgPaint = Paint()..color = Colors.black.withValues(alpha: 0.55);
        canvas.drawRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(4)), bgPaint);
        textPainter.paint(canvas, tp);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _BoxesOnlyPainter oldDelegate) =>
      oldDelegate.results != results || oldDelegate.drawLabels != drawLabels || oldDelegate.screenSize != screenSize;
}


