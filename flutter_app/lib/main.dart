import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

void main() {
  runApp(const DiseasePredictorApp());
}

class DiseasePredictorApp extends StatelessWidget {
  const DiseasePredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Symptom Checker',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00897B),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const SymptomSelectorScreen(),
    );
  }
}

class SymptomSelectorScreen extends StatefulWidget {
  const SymptomSelectorScreen({super.key});

  @override
  State<SymptomSelectorScreen> createState() => _SymptomSelectorScreenState();
}

class _SymptomSelectorScreenState extends State<SymptomSelectorScreen> {
  Interpreter? _interpreter;
  bool _isLoading = true;
  String? _loadError;

  List<String> _rawSymptoms = [];
  List<String> _diseaseLabels = [];
  Map<String, List<String>> _precautionsMap = {};
  final Set<String> _selectedSymptoms = {};

  final TextEditingController _searchController = TextEditingController();
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    _initializeModelAndSchema();
  }

  String _formatSymptomName(String raw) {
    return raw
        .trim()
        .replaceAll('_', ' ')
        .split(' ')
        .where((str) => str.isNotEmpty)
        .map((str) => '${str[0].toUpperCase()}${str.substring(1)}')
        .join(' ');
  }

  Future<void> _initializeModelAndSchema() async {
    try {
      // 1. Load symptom/disease schema
      final schemaJson =
      await rootBundle.loadString('assets/symptom_schema.json');
      final Map<String, dynamic> schema = json.decode(schemaJson);

      _rawSymptoms =
          List<String>.from(schema['symptoms']).map((s) => s.trim()).toList();
      _diseaseLabels =
          List<String>.from(schema['diseases']).map((d) => d.trim()).toList();

      // 2. Load precautions (was previously ignored entirely)
      final precautionsJson =
      await rootBundle.loadString('assets/disease_precautions.json');
      final Map<String, dynamic> rawPrecautions = json.decode(precautionsJson);
      _precautionsMap = rawPrecautions.map(
            (key, value) => MapEntry(key.trim(), List<String>.from(value)),
      );

      // 3. Load TFLite model
      _interpreter =
      await Interpreter.fromAsset('assets/disease_model.tflite');

      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      debugPrint(
          "Loaded ${_rawSymptoms.length} symptoms, ${_diseaseLabels.length} diseases.");
      debugPrint("Input shape: $inputShape, Output shape: $outputShape");

      if (inputShape.last != _rawSymptoms.length) {
        debugPrint(
            "WARNING: model expects ${inputShape.last} inputs but schema has ${_rawSymptoms.length} symptoms. "
                "Retrain and re-export using the same feature list.");
      }

      setState(() => _isLoading = false);
    } catch (e) {
      debugPrint("Initialization Error: $e");
      setState(() {
        _isLoading = false;
        _loadError = e.toString();
      });
    }
  }

  void _toggleSymptom(String rawKey) {
    setState(() {
      if (_selectedSymptoms.contains(rawKey)) {
        _selectedSymptoms.remove(rawKey);
      } else {
        _selectedSymptoms.add(rawKey);
      }
    });
  }

  void _predictDisease() {
    FocusScope.of(context).unfocus();
    if (_interpreter == null || _selectedSymptoms.isEmpty) return;

    try {
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      final inputShape = inputTensor.shape; // [1, numFeatures]
      final outputShape = outputTensor.shape; // [1, numClasses]
      final numInputs = inputShape.last;
      final numOutputs = outputShape.last;

      // Build input vector in exact schema order
      List<double> symptomValues = _rawSymptoms
          .map((s) => _selectedSymptoms.contains(s) ? 1.0 : 0.0)
          .toList();

      if (symptomValues.length < numInputs) {
        symptomValues.addAll(List.filled(numInputs - symptomValues.length, 0.0));
      } else if (symptomValues.length > numInputs) {
        symptomValues = symptomValues.sublist(0, numInputs);
      }

      final input = [Float32List.fromList(symptomValues).reshape([1, numInputs])];

      final outputBuffer = List.filled(numOutputs, 0.0).reshape([1, numOutputs]);
      _interpreter!.run(input, outputBuffer);

      final List<double> probabilities =
      (outputBuffer[0] as List).map((e) => (e as num).toDouble()).toList();

      int maxIndex = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      final diseaseName =
      maxIndex < _diseaseLabels.length ? _diseaseLabels[maxIndex] : 'Unknown';
      final confidence = (maxProb * 100).clamp(0.0, 100.0);

      // Look up precautions straight from the JSON (covers every disease,
      // not just the 5 that used to be hardcoded)
      final precautions = _precautionsMap[diseaseName] ??
          const [
            'Consult a certified healthcare provider for an accurate diagnosis.',
            'Keep track of how your symptoms change over time.',
          ];

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => DiagnosisDetailScreen(
            diseaseName: diseaseName,
            confidenceScore: confidence,
            precautions: precautions,
          ),
        ),
      );
    } catch (e) {
      debugPrint("Inference Error: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Inference failed: $e')),
      );
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Loading model & symptoms...'),
            ],
          ),
        ),
      );
    }

    if (_loadError != null) {
      return Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.error_outline, color: Colors.red, size: 48),
                const SizedBox(height: 12),
                const Text(
                  'Could not load the model or data files.',
                  style: TextStyle(fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  _loadError!,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                const Text(
                  'Check that disease_model.tflite, symptom_schema.json, and '
                      'disease_precautions.json are in assets/ and declared in pubspec.yaml.',
                  style: TextStyle(fontSize: 12),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      );
    }

    final filteredSymptoms = _rawSymptoms
        .where((s) => _formatSymptomName(s)
        .toLowerCase()
        .contains(_searchQuery.toLowerCase()))
        .toList();

    return Scaffold(
      backgroundColor: Colors.grey.shade50,
      appBar: AppBar(
        title: const Text('Disease Symptom Checker',
            style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Theme.of(context).colorScheme.primaryContainer,
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
              child: TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Search symptoms (e.g., headache, fever)...',
                  prefixIcon: const Icon(Icons.search),
                  suffixIcon: _searchQuery.isNotEmpty
                      ? IconButton(
                    icon: const Icon(Icons.clear),
                    onPressed: () {
                      _searchController.clear();
                      setState(() => _searchQuery = '');
                    },
                  )
                      : null,
                  filled: true,
                  fillColor: Colors.white,
                  contentPadding: const EdgeInsets.symmetric(vertical: 0),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide(color: Colors.grey.shade300),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide(color: Colors.grey.shade300),
                  ),
                ),
                onChanged: (val) => setState(() => _searchQuery = val),
              ),
            ),
            if (_selectedSymptoms.isNotEmpty)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                color: Theme.of(context)
                    .colorScheme
                    .primaryContainer
                    .withOpacity(0.3),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Selected Symptoms (${_selectedSymptoms.length})',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ),
                        InkWell(
                          onTap: () => setState(() => _selectedSymptoms.clear()),
                          child: const Text('Clear All',
                              style: TextStyle(fontSize: 12, color: Colors.red)),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    SizedBox(
                      height: 38,
                      child: ListView(
                        scrollDirection: Axis.horizontal,
                        children: _selectedSymptoms.map((rawKey) {
                          return Padding(
                            padding: const EdgeInsets.only(right: 6.0),
                            child: InputChip(
                              label: Text(_formatSymptomName(rawKey),
                                  style: const TextStyle(fontSize: 12)),
                              onDeleted: () => _toggleSymptom(rawKey),
                              deleteIcon: const Icon(Icons.cancel, size: 16),
                              backgroundColor: Colors.white,
                              selected: true,
                              showCheckmark: false,
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                  ],
                ),
              ),
            Expanded(
              child: filteredSymptoms.isEmpty
                  ? Center(
                child: Text('No symptoms match "$_searchQuery"',
                    style: TextStyle(color: Colors.grey.shade600)),
              )
                  : ListView.separated(
                padding: const EdgeInsets.symmetric(vertical: 4),
                itemCount: filteredSymptoms.length,
                separatorBuilder: (context, index) =>
                    Divider(height: 1, color: Colors.grey.shade200),
                itemBuilder: (context, index) {
                  final rawKey = filteredSymptoms[index];
                  final isChecked = _selectedSymptoms.contains(rawKey);
                  return CheckboxListTile(
                    value: isChecked,
                    title: Text(
                      _formatSymptomName(rawKey),
                      style: TextStyle(
                        fontWeight:
                        isChecked ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                    activeColor: Theme.of(context).colorScheme.primary,
                    onChanged: (_) => _toggleSymptom(rawKey),
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: SizedBox(
                width: double.infinity,
                height: 52,
                child: ElevatedButton.icon(
                  onPressed: _selectedSymptoms.isNotEmpty ? _predictDisease : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Theme.of(context).colorScheme.primary,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12)),
                    elevation: 2,
                  ),
                  icon: const Icon(Icons.analytics_outlined),
                  label: Text(
                    _selectedSymptoms.isEmpty
                        ? 'Select symptoms to analyze'
                        : 'Analyze ${_selectedSymptoms.length} Symptom${_selectedSymptoms.length > 1 ? 's' : ''}',
                    style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class DiagnosisDetailScreen extends StatelessWidget {
  final String diseaseName;
  final double confidenceScore;
  final List<String> precautions;

  const DiagnosisDetailScreen({
    super.key,
    required this.diseaseName,
    required this.confidenceScore,
    required this.precautions,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis Details'),
        backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Card(
              elevation: 2,
              shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      diseaseName,
                      style:
                      const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Model Confidence: ${confidenceScore.toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade700,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: (confidenceScore / 100).clamp(0.0, 1.0),
                        minHeight: 6,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            const Text('Recommended Precautions & Care',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            ...precautions.map(
                  (step) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.check_circle_outline, size: 20, color: Colors.teal),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(step, style: const TextStyle(fontSize: 14, height: 1.3)),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            Center(
              child: Text(
                'Disclaimer: AI predictions are for informational purposes only and do not replace professional medical advice.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
              ),
            ),
          ],
        ),
      ),
    );
  }
}